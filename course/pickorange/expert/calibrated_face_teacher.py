"""Privileged teacher: calibrated center, face grasp, measured gates.

Native servos/physics/success are unchanged. Mesh proxies are explicitly not
contact sensors. Only bounded six-joint commands leave this teacher.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from diameter_teacher import DiameterTeacher, DiameterKinematics
from geometric_teacher import GeometricTeacher
from finger_geometry import FingerGeometry
from grasp_quality import check_grasp


def measured_alignment(fixed, moving, center, desired):
    fixed,moving,center,desired=map(lambda v:np.asarray(v,dtype=float),(fixed,moving,center,desired))
    line=moving-fixed
    gap=float(np.linalg.norm(line))
    if not np.isfinite(np.r_[fixed,moving,center,desired]).all() or gap<1e-6:
        raise ValueError('Invalid pad geometry')
    axis=line/gap
    offset=center-(fixed+moving)/2
    axial=float(offset@axis)
    return dict(gap_m=gap,center_error_m=float(np.linalg.norm(offset)),
        diameter_line_error_m=float(np.linalg.norm(offset-axial*axis)),
        diameter_axis_error=float(np.linalg.norm(axis-desired)),
        finger_height_difference_m=float(abs(line[2])),
        object_between_fingers=bool(abs(axial)<gap/2))


def tight_gate(metrics, diameter, angles, sent, target):
    """Closure admission, NOT success. A command alone cannot pass."""
    angles=np.asarray(angles,dtype=float)
    if len(angles)<12 or not np.isfinite(angles).all(): return False
    return bool(abs(sent-target)<1e-4 and np.ptp(angles[-12:])<=.025
        and .70*diameter<=metrics['gap_m']<=1.12*diameter
        and metrics['center_error_m']<.008 and metrics['diameter_line_error_m']<.006
        and metrics['diameter_axis_error']<.14 and metrics['finger_height_difference_m']<.008
        and metrics['object_between_fingers'])


class CalibratedFaceTeacher(DiameterTeacher):
    caps=(240,220,180,180,180,100,80,90,250)

    def __init__(self,asset,close_rad=.2):
        self.kin=DiameterKinematics(asset)
        self.geometry=FingerGeometry(asset)
        self.close_rad=close_rad
        self.calibration=json.loads(Path(__file__).with_name('object_center_calibration.json').read_text())
        self.scan=json.loads(Path(__file__).with_name('finger_geometry_calibration.json').read_text())
        self.table_z=max(v['bounds'][1][2] for v in self.calibration['support_candidates'])

    def root_object(self):
        return self.env.scene[f'Orange00{self._orange_now}'].data.root_pos_w[0].cpu().numpy().copy()

    def center_for(self,name):
        obj=self.env.scene[name].data
        quat=obj.root_quat_w[0].cpu().numpy()
        r=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
        return obj.root_pos_w[0].cpu().numpy()+r@np.asarray(self.calibration['objects'][name]['center_offset_root_m'])

    def object(self):
        return self.center_for(f'Orange00{self._orange_now}')

    def choose_face(self):
        local=self.rotation.T@(self.object()-self.base)
        radial=local[:2]/np.linalg.norm(local[:2])
        reference=self.measured()[:5]
        extent=self.calibration['objects'][f'Orange00{self._orange_now}']['half_extent_m']
        self.object_diameter=2*float(np.linalg.norm(np.asarray(extent)[:2]))/np.sqrt(2)
        candidates=[]
        fallbacks=[]
        # Offline templates are recalculated at the actual settled center;
        # they are not static joint commands or a filtered initial distribution.
        templates=[r for r in self.scan['valid_candidates'] if r['orange']==self._orange_now]
        plate=self.env.scene['Plate'].data
        plate_pos=plate.root_pos_w[0].cpu().numpy()
        plate_quat=plate.root_quat_w[0].cpu().numpy()
        plate_rotation=Rotation.from_quat(plate_quat[[1,2,3,0]]).as_matrix()
        plate_bounds=np.asarray(self.calibration['objects']['Plate']['local_bounds'])
        plate_radius=float(max(self.calibration['objects']['Plate']['half_extent_m'][:2]))
        near_plate=np.linalg.norm((self.object()-plate_pos)[:2])<plate_radius+.07
        if near_plate:
            toward=self.rotation.T@(plate_pos-self.object())
            tangent_yaw=np.arctan2(toward[1],toward[0])+np.pi/2
            base_yaw=np.arctan2(radial[1],radial[0])+np.pi/2
            angle=np.rad2deg(tangent_yaw-base_yaw)
            templates=[dict(row,azimuth_offset_deg=float(a)) for row in templates
                       for a in (angle,angle-180,angle+180,-60,-45,-30,-15,15,30,40,45,60)]+templates
        templates.sort(key=lambda r:(0 if near_plate and abs(r['azimuth_offset_deg'])>35 else 1,
                       .1*abs(r['opening_rad']-.75)+abs(r['depth_m']+.095)*3
                       +abs(r['azimuth_offset_deg'])/1000))
        seen=set()
        tip=min(float(v['canonical'][:,2].min()) for v in self.geometry.meshes['gripper'])
        for row in templates:
            if hasattr(self,'diagnostic_out'):
                (self.diagnostic_out/'planning.json').write_text(json.dumps(dict(orange=self._orange_now,candidate=row,accepted=len(candidates))))
            if row['depth_m']-tip<.001: continue
            key=(row['depth_m'],row['opening_rad'],row['azimuth_offset_deg'])
            if key in seen: continue
            seen.add(key)
            self.kin.fixed,self.kin.moving=self.geometry.pad_pair(row['depth_m'])
            self.kin.set_opening(row['opening_rad'])
            yaw=np.arctan2(radial[1],radial[0])+np.pi/2+np.deg2rad(row['azimuth_offset_deg'])
            axis=np.array([np.cos(yaw),np.sin(yaw),0.])
            self.kin.desired_axis=axis
            grasp=self.kin.solve(local,reference)
            approach=self.kin.solve(local+np.array([0,0,.080]),reference)
            fallbacks.append(dict(template=row,axis=axis,
                cost=max(grasp['position_error_m'],approach['position_error_m'])+.04*max(grasp['tool_axis_error'],approach['tool_axis_error']),
                clearance=dict(approximate_plan=True),contact_rad=min(row['opening_rad'],.6)))
            if max(grasp['position_error_m'],approach['position_error_m'])>.004 or \
                    max(grasp['tool_axis_error'],approach['tool_axis_error'])>.12: continue
            # A held orange prevents empty-close q=.2. Check the geometry at
            # estimated contact opening, not an impossible interpenetrating pose.
            gaps=[]
            for angle in np.linspace(.2,row['opening_rad'],31):
                moving=(self.geometry.jaw_transform(angle)@np.r_[self.kin.moving,1])[:3]
                gaps.append((abs(np.linalg.norm(moving-self.kin.fixed)-self.object_diameter),angle))
            contact_rad=min(gaps)[1]
            closure_reachable=True
            for angle in (contact_rad,min(row['opening_rad'],contact_rad+.03)):
                self.kin.set_opening(angle)
                sol=self.kin.solve(local,grasp['q'])
                if sol['position_error_m']>.004 or sol['tool_axis_error']>.12:
                    closure_reachable=False;break
            self.kin.set_opening(row['opening_rad'])
            if not closure_reachable: continue
            t,_,_=self.kin.fk(grasp['q'])
            xf=np.eye(4);xf[:3,:3]=self.rotation@t[:3,:3];xf[:3,3]=self.base+self.rotation@t[:3,3]
            extent=self.calibration['objects'][f'Orange00{self._orange_now}']['half_extent_m']
            clear=self.geometry.clearance(xf,row['opening_rad'],self.object(),extent,self.table_z)
            if clear['table_clearance_m']<.002 or clear['sphere_plane_margin_m']<.001: continue
            # Check descent joint interpolation against table and all oranges.
            path_clear=True
            plate_margin=np.inf
            for fraction in np.linspace(0,1,9):
                t,_,_=self.kin.fk((1-fraction)*approach['q']+fraction*grasp['q'])
                xf[:3,:3]=self.rotation@t[:3,:3];xf[:3,3]=self.base+self.rotation@t[:3,3]
                for _,points in self.geometry.transformed_points(xf,row['opening_rad']):
                    relative=(points-plate_pos)@plate_rotation
                    low=relative[(relative[:,2]<plate_bounds[1,2]+.002)&(relative[:,2]>plate_bounds[0,2]-.002)]
                    if len(low):plate_margin=min(plate_margin,float(np.linalg.norm(low[:,:2],axis=1).min()-plate_radius))
                for i in (1,2,3):
                    name=f'Orange00{i}'
                    c=self.geometry.clearance(xf,row['opening_rad'],self.center_for(name),
                        self.calibration['objects'][name]['half_extent_m'],self.table_z)
                    if c['table_clearance_m']<.002 or c['sphere_plane_margin_m']<.0005:
                        path_clear=False;break
                if not path_clear: break
            if not path_clear: continue
            cost=grasp['position_error_m']+.04*grasp['tool_axis_error']+.002*np.linalg.norm(approach['q']-reference)
            cost+=.5*max(0.,.008-(row['depth_m']-tip))
            cost+=10*max(0.,.002-plate_margin)
            clear['plate_cylinder_margin_m']=float(plate_margin) if np.isfinite(plate_margin) else None
            candidates.append(dict(template=row,axis=axis,cost=cost,clearance=clear,contact_rad=contact_rad))
            if len(candidates)>=3 and plate_margin>=.002: break
        assert candidates or fallbacks, 'No finite pad/IK candidate'
        best=min(candidates or fallbacks,key=lambda v:v['cost'])
        row=best['template']
        self.kin.fixed,self.kin.moving=self.geometry.pad_pair(row['depth_m'])
        self.kin.desired_axis=best['axis']
        self.open_rad=row['opening_rad']
        self.expected_contact_rad=float(best['contact_rad'])
        self.events.append(dict(calibrated_face_plan=dict(orange=self._orange_now,
            center_world_m=self.object().tolist(),depth_local_m=row['depth_m'],
            fixed_finger_insertion_from_tip_m=row['depth_m']-tip,opening_rad=self.open_rad,
            diameter_estimate_m=self.object_diameter,azimuth_offset_deg=row['azimuth_offset_deg'],
            expected_contact_opening_rad=self.expected_contact_rad,
            clearance_proxy=best['clearance'],candidates_accepted=len(candidates),
            collision_sensor_verified=False)))

    def enter(self,stage):
        if stage==0: self.choose_face()
        elif stage==4: self.kin.desired_axis=None
        self.kin.set_opening(self.open_rad if stage in (0,1) else float(self.measured()[5]))
        GeometricTeacher.enter(self,stage)
        if stage==1:
            self.descent_waypoint=self.tcp().copy()
        elif stage==2:
            self.close_anchor=self.object().copy()
            self.grasp_root_z=float(self.root_object()[2])
        elif stage==3:
            self.micro_lift_verified=False
            self.micro_height,self.micro_relative=[],[]
            self.full_lift_target=self.target.copy()
            self.target=self.pick_anchor+np.array([0,0,.015])

    def pad_metrics(self):
        self.kin.set_opening(float(self.measured()[5]))
        quat=self.robot.data.body_quat_w[0,self.body_id].cpu().numpy()
        r=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
        origin=self.robot.data.body_pos_w[0,self.body_id].cpu().numpy()
        # Jaw transform already includes both joint local frames.
        moving=(self.geometry.jaw_transform(float(self.measured()[5]))@np.r_[self.kin.moving,1])[:3]
        desired=self.rotation@self.kin.desired_axis
        return measured_alignment(origin+r@self.kin.fixed,origin+r@moving,self.object(),desired)

    def command(self,previous,measured):
        self.kin.set_opening(float(measured[5]))
        if self.stage in (0,1,2,3,4,5,7):
            target=self.target.copy()
            if self.stage==1:
                full=self.object()
                if np.linalg.norm(self.tcp()-self.descent_waypoint)<.006:
                    delta=full-self.descent_waypoint
                    self.descent_waypoint+=delta*min(1.,.002/max(np.linalg.norm(delta),1e-9))
                target=self.descent_waypoint
            elif self.stage==2:
                target=self.close_anchor.copy()  # do not chase a pushed orange
            solved=self.kin.solve(self.rotation.T@(target-self.base),measured[:5],local=True)
            assert np.isfinite(solved['q']).all()
            self.goal=solved['q']
        hold=getattr(self,'close_hold_command',None)
        if hold is not None: del self.close_hold_command
        old_integral=self.integral.copy()
        wanted=GeometricTeacher.command(self,previous,measured)
        if self.stage in (1,2) and abs(previous[:5]-measured[:5]).max()>.12:
            # Anti-windup: preserve loaded bias, don't keep integrating contact.
            self.integral=old_integral
            wanted[:5]=self.goal+self.gravity_offset+self.integral
        if hold is not None: self.close_hold_command=hold
        wanted[5]=(-np.pi/18 if self.stage==8 else
            self.close_rad if self.stage in (2,3,4,5) else self.open_rad)
        arm_step=.008 if self.stage in (1,2) else .030
        wanted[:5]=previous[:5]+np.clip(wanted[:5]-previous[:5],-arm_step,arm_step)
        grip_step=.012 if self.stage==2 else .030
        wanted[5]=previous[5]+np.clip(wanted[5]-previous[5],-grip_step,grip_step)
        return wanted

    def observe(self):
        if self.stage not in (1,2,3):
            return GeometricTeacher.observe(self)
        self.elapsed+=1
        metrics=self.pad_metrics()
        if self.stage==1:
            aligned=(metrics['center_error_m']<.006 and metrics['diameter_line_error_m']<.006
                and metrics['diameter_axis_error']<.14 and metrics['finger_height_difference_m']<.008
                and metrics['object_between_fingers'])
            self.stable=self.stable+1 if aligned else 0
            if self.stable>=6:
                self.events.append(dict(completed_stage='descend',orange=self._orange_now,metrics=metrics))
                self.enter(2);return
        elif self.stage==2:
            self.angles.append(float(self.measured()[5]))
            if tight_gate(metrics,self.object_diameter,self.angles,float(self.applied_command[5]),self.close_rad):
                self.closure_reports[self._orange_now]=dict(verified=True,**metrics,
                    target_rad=self.close_rad,measured_gripper_rad=float(self.measured()[5]),
                    contact_sensor_verified=False)
                self.lift_initial_z=float(self.object()[2])
                self.events.append(dict(completed_stage='close',orange=self._orange_now,metrics=metrics))
                self.enter(3);return
        else:
            height=float(self.root_object()[2])
            relative=self.object()-self.tcp()
            if not self.micro_lift_verified:
                self.micro_height.append(height)
                self.micro_relative.append(relative.copy())
                if len(self.micro_height)>=12:
                    drift=float(np.linalg.norm(np.asarray(self.micro_relative[-12:])-relative,axis=1).max())
                    lift=float(min(self.micro_height[-12:])-self.grasp_root_z)
                    if lift>=.007 and drift<.006 and np.linalg.norm(self.tcp()-self.target)<.008:
                        self.micro_lift_verified=True
                        self.events.append(dict(micro_lift_verified=True,orange=self._orange_now,
                            minimum_root_lift_m=lift,relative_drift_m=drift))
                        self.target=self.full_lift_target.copy()
                        self.heights,self.relatives,self.angles=[],[],[]
            else:
                self.heights.append(height);self.relatives.append(relative.copy())
                self.angles.append(float(self.measured()[5]))
                checked=check_grasp(self.relatives,self.heights,self.angles,self.grasp_root_z)
                if checked['verified'] and np.linalg.norm(self.tcp()-self.target)<.012:
                    self.grasp_reports[self._orange_now]=checked
                    self.relative_reference=relative.copy()
                    self.events.append(dict(completed_stage='lift',orange=self._orange_now,grasp=checked))
                    self.enter(4);return
        if self.elapsed>=self.caps[self.stage]:
            self.events.append(dict(calibrated_failure_metrics=metrics))
            self.fail('calibrated face stage deadline; measured admission not met')
