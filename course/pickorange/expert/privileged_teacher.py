"""Privileged planning and measured grasp feedback, never actor observations."""
import numpy as np
from conservative_efficient_teacher import ConservativeEfficientTeacher
from calibrated_face_teacher import CalibratedFaceTeacher
from geometric_teacher import GeometricTeacher


class PrivilegedTeacher(ConservativeEfficientTeacher):
    # Phase timing is a planning heuristic; the environment controls the rollout horizon.
    caps = (2701,) * 9

    def fail(self, reason):
        if self.stage in (4,5):
            plate=self.env.scene['Plate'].data.root_pos_w[0].cpu().numpy()
            delta=self.root_object()-plate
            if np.all(np.abs(delta[:2])<.1) and -.07<delta[2]<.07:
                self.events.append(dict(released_into_plate_during_transport=True,orange=self._orange_now))
                self.goal=self.measured()[:5].copy()
                self.integral[:]=0.
                self.enter(6)
                return
        # Planning proxies request recovery, never terminate the episode.
        self.events.append(dict(recovery_requested=reason,stage=self.stages[self.stage]))
        self.recover_pending=True

    def reset(self):
        super().reset()
        self.recover_pending=False

    def enter(self,stage):
        if stage!=3:
            super().enter(stage)
            if stage==1:
                self.descent_q_history=[]
            elif stage==4:
                self.plan_placement()
                self.target=self.release_tcp.copy()
                obstacle_top=max(self.center_for(f'Orange00{i}')[2]+self.calibration['objects'][f'Orange00{i}']['half_extent_m'][2] for i in (1,2,3) if i!=self._orange_now)
                self.target[2]=max(self.target[2]+.07,self.tcp()[2],obstacle_top+.06)
                self.goal=self.kin.solve(self.rotation.T@(self.target-self.base),self.measured()[:5])['q']
                self.carry_start=self.tcp().copy()
                self.carry_waypoint=self.carry_start.copy();self.carry_waypoint[2]=self.target[2]
                self.carry_horizontal=False
                transform,_,_=self.kin.fk(self.measured()[:5])
                self.carry_axis_start=transform[:3,:3]@self.kin.pad_axis
                radial=(self.rotation.T@(self.target-self.base))[:2]
                radial=radial/np.linalg.norm(radial)
                self.carry_axis_end=np.r_[-radial[1],radial[0],0.]
                self.carry_delta=np.zeros(5)
            elif stage==5:
                self.placement_stable=0
                self.placement_positions=[]
                self.target=self.release_tcp.copy()
                self.goal=self.kin.solve(self.rotation.T@(self.target-self.base),self.measured()[:5])['q']
            elif stage==6:
                self.release_withdraw=None
                self.release_positions=[]
            return
        self.stage,self.elapsed,self.stable=3,0,0
        self.angles,self.heights,self.relatives,self.joint_history=[],[],[],[]
        self.micro_lift_verified=False
        self.micro_height,self.micro_relative=[],[]
        radial=self.pick_anchor[:2]-self.base[:2]
        radial=radial/np.linalg.norm(radial)
        self.kin.set_opening(float(self.measured()[5]))
        start=self.tcp().copy()
        self.grasp_root_z=float(self.root_object()[2])
        transform,_,_=self.kin.fk(self.measured()[:5])
        self.lift_up=transform[:3,2].copy()
        self.kin.desired_axis=transform[:3,:3]@self.kin.pad_axis
        self.target=start+np.r_[-.005*radial,.015]
        self.full_lift_target=start+np.r_[-.015*radial,.05]
        solution=self.kin.solve(self.rotation.T@(self.target-self.base),self.measured()[:5],local=True)
        self.goal=solution['q']
        self.events.append(dict(stage='lift',orange=self._orange_now,inward_lift_all_objects=True,
            micro_target=self.target.tolist(),full_target=self.full_lift_target.tolist(),
            initial_ik_position_error=solution['position_error_m']))

    def plan_placement(self):
        """Use live plate surface, occupied volumes and held-object offset."""
        from scipy.spatial.transform import Rotation
        from omni.physx import get_physx_scene_query_interface
        import carb
        plate=self.env.scene['Plate'].data
        origin=plate.root_pos_w[0].cpu().numpy()
        quat=plate.root_quat_w[0].cpu().numpy()
        rot=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
        half=np.asarray(self.calibration['objects'][f'Orange00{self._orange_now}']['half_extent_m'])
        objquat=self.env.scene[f'Orange00{self._orange_now}'].data.root_quat_w[0].cpu().numpy()
        objrot=Rotation.from_quat(objquat[[1,2,3,0]]).as_matrix()
        vertical=float(np.linalg.norm(objrot[2]*half))
        radius=float(max(self.calibration['objects']['Plate']['half_extent_m'][:2]))
        others=[(self.center_for(f'Orange00{i}'),np.asarray(self.calibration['objects'][f'Orange00{i}']['half_extent_m']))
                for i in (1,2,3) if i!=self._orange_now]
        candidates=[]
        query=get_physx_scene_query_interface()
        for radius_fraction in (.35,.50,0.):
            for angle in np.linspace(0,2*np.pi,12,endpoint=False):
                xy=(origin+rot@np.array([radius*radius_fraction*np.cos(angle),radius*radius_fraction*np.sin(angle),0.]))[:2]
                spacing=min(float(np.linalg.norm(xy-center[:2])-max(extent[:2])-max(half[:2])) for center,extent in others)
                hit=query.raycast_closest(carb.Float3(float(xy[0]),float(xy[1]),float(origin[2]+.3)),carb.Float3(0,0,-1),.5)
                if not hit['hit'] or 'Plate' not in hit['rigidBody']:continue
                surface=float(hit['position'][2])
                # The near rim is reachable but sloped: fruit rolls out on release.
                # Prefer low, flat plate surfaces rather than distance alone.
                flat=float(np.asarray(hit['normal'])@rot[:,2])
                candidates.append((max(0.,.01-spacing)*10+.1*np.linalg.norm(xy-self.base[:2])
                    +.3*max(0.,1-flat)+.5*(surface-origin[2]),xy,surface,spacing))
        assert candidates, 'No visible plate surface for privileged placement planning'
        offset=self.object()-self.tcp()
        self.kin.set_opening(float(self.measured()[5]))
        plans=[]
        for _,xy,surface,spacing in sorted(candidates,key=lambda x:x[0])[:4]:
            target=np.r_[xy,surface+vertical+.004]-offset
            for _ in range(3):
                solution=self.kin.solve(self.rotation.T@(target-self.base),self.measured()[:5])
                t,_,_=self.kin.fk(solution['q'])
                xf=np.eye(4);xf[:3,:3]=self.rotation@t[:3,:3];xf[:3,3]=self.base+self.rotation@t[:3,3]
                points=np.concatenate([p for _,p in self.geometry.transformed_points(xf,float(self.measured()[5]))])
                clearance=float(points[:,2].min()-surface)
                raise_by=max(0.,.012-clearance)
                # Query the rim and occupied surfaces below the low fingers too.
                for point in points[points[:,2]<surface+.06][::3]:
                    hit=query.raycast_closest(carb.Float3(float(point[0]),float(point[1]),float(origin[2]+.3)),carb.Float3(0,0,-1),.5)
                    if hit['hit'] and ('Plate' in hit['rigidBody'] or 'Orange' in hit['rigidBody']):
                        raise_by=max(raise_by,float(hit['position'][2])+.008-point[2])
                if raise_by<=1e-4:break
                target[2]+=raise_by
            margins=[self.geometry.clearance(xf,float(self.measured()[5]),c,e,self.table_z)['sphere_plane_margin_m'] for c,e in others]
            score=solution['position_error_m']+max(0.,.006-min(margins))*10+max(0.,.008-spacing)*10+.05*(target[2]-surface)
            plans.append((score,target,dict(surface_z=surface,orange_spacing_m=spacing,finger_surface_clearance_m=clearance,
                other_orange_margin_m=min(margins),object_half_height_m=vertical,target_tcp=target.tolist())))
        _,self.release_tcp,evidence=min(plans,key=lambda x:x[0])
        self.placement_surface=evidence['surface_z']
        self.placement_plate_z=float(origin[2])
        self.placement_object_xy=self.release_tcp[:2]+offset[:2]
        self.drop_anchor=self.release_tcp.copy();self.drop_anchor[2]-=.07
        self.events.append(dict(privileged_placement=evidence))

    def command(self, previous, measured):
        if self.recover_pending:
            self.recover_pending=False
            self.integral[:]=0.
            self.enter(0)
        # Keep the closure anchor fixed while compensating the moving jaw.
        # Chasing a displaced orange during closure can push it repeatedly.
        # Actual failed lifts still replan from its latest privileged position.
        if self.stage == 1:
            # Execute the complete grasp IK goal chosen at stage entry.
            # Re-solving a 6 mm lookahead from a loaded/oscillating measured
            # pose kept moving the goal before the native servo could track it.
            wanted=GeometricTeacher.command(self,previous,measured)
            wanted[:5]=previous[:5]+np.clip(wanted[:5]-previous[:5],-.015,.015)
            wanted[5]=previous[5]+np.clip(self.open_rad-previous[5],-.03,.03)
            return wanted
        if self.stage == 2:
            # A unilateral jaw should pinch against its stationary finger.
            # Re-solving midpoint/axis during closure rotated the whole hand.
            wanted=previous.copy()
            wanted[:5]=self.close_hold_command
            # Close free space promptly; slow before the estimated fruit surface.
            # The measured-command error cap below still limits contact force.
            step=.004 if self.pad_metrics()['gap_m']<1.08*self.object_diameter else .012
            wanted[5]=previous[5]+np.clip(self.close_rad-previous[5],-step,step)
            # Do not accumulate a large closing error against a contacted fruit.
            wanted[5]=max(wanted[5],measured[5]-.035)
            return wanted
        if self.stage==3:
            # Track a short Cartesian path with the nearest joint solution.
            # A fixed tool-up objective overconstrained this 5-DOF arm.
            from scipy.optimize import least_squares
            self.kin.set_opening(float(measured[5]))
            actual=self.tcp()
            delta=self.target-actual
            waypoint=actual+delta*min(1.,.006/max(np.linalg.norm(delta),1e-9))
            target=self.rotation.T@(waypoint-self.base)
            def residual(q):
                transform,point,_=self.kin.fk(q)
                return np.r_[point-target,.005*(q-measured[:5])]
            self.goal=least_squares(residual,np.clip(measured[:5],self.kin.limits[:,0]+1e-4,self.kin.limits[:,1]-1e-4),
                bounds=(self.kin.limits[:,0]+1e-4,self.kin.limits[:,1]-1e-4),max_nfev=40).x
            wanted=GeometricTeacher.command(self,previous,measured)
            delta=wanted[:5]-previous[:5]
            wanted[:5]=previous[:5]+delta*min(1.,.012/max(np.max(np.abs(delta)),1e-9))
            wanted[5]=self.close_rad
            return wanted
        if self.stage==4:
            # Raise above occupied volumes first; then follow a Cartesian path.
            # One far joint-space goal can dip through the occupied plate en route.
            actual=self.tcp()
            self.carry_horizontal=self.carry_horizontal or actual[2]>=self.target[2]-.012
            if self.carry_horizontal:
                route=self.target[:2]-self.carry_start[:2]
                length=max(np.linalg.norm(route),1e-9)
                direction=route/length
                along=np.clip((actual[:2]-self.carry_start[:2])@direction+.03,0.,length)
                self.carry_waypoint[:2]=self.carry_start[:2]+along*direction
                if actual[2]<self.target[2]-.02:
                    self.carry_waypoint[:2]=actual[:2]
            fraction=min(1.,np.linalg.norm(self.carry_waypoint[:2]-self.carry_start[:2])/
                max(np.linalg.norm(self.target[:2]-self.carry_start[:2]),1e-9))
            axis=(1-fraction)*self.carry_axis_start+fraction*self.carry_axis_end
            self.kin.desired_axis=axis/max(np.linalg.norm(axis),1e-9)
            self.kin.set_opening(float(measured[5]))
            self.goal=self.kin.solve(self.rotation.T@(self.carry_waypoint-self.base),measured[:5],local=True)['q']
            wanted=GeometricTeacher.command(self,previous,measured)
            delta=np.clip(wanted[:5]-previous[:5],-.03,.03)
            self.carry_delta+=np.clip(delta-self.carry_delta,-.006,.006)
            wanted[:5]=previous[:5]+self.carry_delta
            wanted[5]=self.close_rad
            return wanted
        if self.stage==6 and measured[5]>=self.open_rad-.03:
            # Low placement naturally leaves the open fingers near the fruit.
            # Withdraw before requiring separation, otherwise both wait forever.
            if self.release_withdraw is None:
                self.release_withdraw=self.tcp()+np.array([0.,0.,.03])
                self.events.append(dict(release_withdraw_up_m=.03,orange=self._orange_now))
            self.kin.set_opening(float(measured[5]))
            self.goal=self.kin.solve(self.rotation.T@(self.release_withdraw-self.base),measured[:5],local=True)['q']
            wanted=GeometricTeacher.command(self,previous,measured)
            wanted[:5]=previous[:5]+np.clip(wanted[:5]-previous[:5],-.008,.008)
            wanted[5]=previous[5]+np.clip(self.open_rad-previous[5],-.03,.03)
            return wanted
        return super().command(previous, measured)

    def observe(self):
        if self.stage == 1:
            self.elapsed+=1
            metrics=self.pad_metrics()
            self.descent_q_history.append(self.measured()[:5].copy())
            self.descent_q_history=self.descent_q_history[-12:]
            if self.elapsed>=30 and np.linalg.norm(self.object()-self.target)>.3*self.object_diameter:
                # A displaced fruit needs a fresh approach, not an episode abort
                # or an indefinite wait at its previous location.
                self.events.append(dict(replan_displaced_during_descent=True,orange=self._orange_now,
                    displacement_m=float(np.linalg.norm(self.object()-self.target))))
                self.enter(0)
                return
            near=(metrics['object_between_fingers'] and
                  metrics['center_error_m']<.16*self.object_diameter and
                  abs(self.tcp()[2]-self.object()[2])<.12*self.object_diameter)
            contact_limited=False
            if not near and len(self.descent_q_history)==12 and np.ptp(
                    np.asarray(self.descent_q_history),axis=0).max()<.01:
                from scipy.spatial.transform import Rotation
                quat=self.env.scene[f'Orange00{self._orange_now}'].data.root_quat_w[0].cpu().numpy()
                rotation=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
                half=np.asarray(self.calibration['objects'][f'Orange00{self._orange_now}']['half_extent_m'])
                relative=rotation.T@(self.tcp()-self.object())
                inside=float(np.linalg.norm(relative/half))
                contact_limited=bool(metrics['object_between_fingers'] and inside<.8)
                if contact_limited:
                    self.events.append(dict(contact_limited_closure_attempt=True,
                        orange=self._orange_now,normalized_ellipsoid_offset=inside,
                        unload_integral=True))
            self.stable=self.stable+1 if near else 0
            if self.stable>=6 or contact_limited:
                self.events.append(dict(stage='descend',close_attempt=True,metrics=metrics))
                self.enter(2)
                if contact_limited:
                    self.goal=self.measured()[:5].copy()
                    self.integral[:]=0.
                    self.close_hold_command=self.goal+self.gravity_offset
            return
        if self.stage == 2:
            self.elapsed += 1
            q = float(self.measured()[5])
            self.angles.append(q)
            metrics = self.pad_metrics()
            # Contact produces a measured ~0.0053 rad periodic oscillation.
            # Compare window means to reject ramping without rejecting chatter.
            stable = (len(self.angles) >= 12 and np.ptp(self.angles[-12:]) < .008
                and abs(np.mean(self.angles[-6:])-np.mean(self.angles[-12:-6])) < .0015)
            blocked = q-float(self.applied_command[5]) > .025
            geometric_contact=bool(metrics['gap_m']<=self.object_diameter
                and metrics['center_error_m']<.2*self.object_diameter
                and metrics['diameter_line_error_m']<.2*self.object_diameter)
            if stable and q<.23 and not blocked:
                self.events.append(dict(replan_after_empty_closure=True,orange=self._orange_now))
                self.enter(0)
                return
            if stable and blocked and metrics['object_between_fingers']:
                # Contact proxy only authorizes a micro-lift. Actual object motion
                # below decides whether to transport; no 6 mm rejection is used.
                self.close_rad = max(.2, q-.025)
                self.adaptive_latched = True
                self.lift_initial_z = float(self.object()[2])
                self.events.append(dict(stage='close',micro_lift_attempt=True,geometric_contact_proxy=geometric_contact,
                    blocked_angle_proxy=blocked,overclosure_rad=.025,metrics=metrics))
                self.enter(3)
            return
        if self.stage == 3:
            self.elapsed += 1
            height = float(self.root_object()[2])
            relative = self.object()-self.tcp()
            self.micro_height.append(height);self.micro_relative.append(relative.copy())
            if len(self.micro_height) < 12:
                return
            lift = float(min(self.micro_height[-12:])-self.grasp_root_z)
            drift = float(np.linalg.norm(np.asarray(self.micro_relative[-12:])-relative,axis=1).max())
            if not self.micro_lift_verified and (lift >= .007 or (self.elapsed>=90 and lift>=.003)) and drift < .01:
                self.micro_lift_verified = True
                self.target = self.full_lift_target.copy()
                self.events.append(dict(physical_micro_lift=True,lift_m=lift,relative_drift_m=drift))
            if self.micro_lift_verified and lift >= .03 and drift < .025 and np.linalg.norm(self.tcp()-self.target)<.02:
                self.relative_reference = relative.copy()
                self.grasp_reports[self._orange_now] = dict(verified=True,lift_m=lift,relative_drift_m=drift)
                self.events.append(dict(completed_stage='lift',orange=self._orange_now,lift_m=lift))
                self.enter(4)
            elif self.elapsed>=120 and lift<.003:
                self.events.append(dict(replan_after_empty_lift=True,orange=self._orange_now,lift_m=lift))
                self.enter(0)
            return
        if self.stage==5:
            self.elapsed+=1
            from scipy.spatial.transform import Rotation
            obj=self.env.scene[f'Orange00{self._orange_now}'].data
            plate=self.env.scene['Plate'].data.root_pos_w[0].cpu().numpy()
            delta=self.root_object()-plate
            quat=obj.root_quat_w[0].cpu().numpy()
            rotation=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
            half=np.asarray(self.calibration['objects'][f'Orange00{self._orange_now}']['half_extent_m'])
            bottom=self.object()[2]-float(np.linalg.norm(rotation[2]*half))
            surface=self.placement_surface+float(plate[2])-self.placement_plate_z
            self.placement_positions.append(self.object().copy())
            settled=(len(self.placement_positions)>=6 and
                np.linalg.norm(np.asarray(self.placement_positions[-6:])-self.object(),axis=1).max()<.003)
            ready=(np.all(np.abs(delta[:2])<.1) and -.07<delta[2]<.07
                and np.linalg.norm(self.object()[:2]-self.placement_object_xy)<.025
                and bottom-surface<=.025
                and settled)
            self.placement_stable=self.placement_stable+1 if ready else 0
            if self.placement_stable>=6:
                self.events.append(dict(low_placement_release=True,orange=self._orange_now,bottom_surface_gap_m=float(bottom-surface)))
                self.enter(6)
            return  # Do not fall through to the old TCP-distance-only release.
        if self.stage == 6:
            self.elapsed += 1
            plate=self.env.scene['Plate'].data.root_pos_w[0].cpu().numpy()
            delta=self.root_object()-plate
            placed=bool(np.all(np.abs(delta[:2])<.1) and -.07<delta[2]<.07)
            self.release_positions.append(self.object().copy())
            placed=placed and len(self.release_positions)>=6 and np.linalg.norm(
                np.asarray(self.release_positions[-6:])-self.object(),axis=1).max()<.003
            released=float(self.measured()[5])>=self.open_rad-.03 and np.linalg.norm(self.object()-self.tcp())>.025
            self.stable=self.stable+1 if placed and released else 0
            if self.stable>=12:
                self.events.append(dict(completed_stage='release',orange=self._orange_now,actual_release_verified=True))
                self.enter(7)
            return
        super().observe()
