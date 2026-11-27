"""Synthetic CPU checks; neither physics nor calibration accuracy is established."""
import importlib.util
from pathlib import Path
import sys
import unittest
import io
import json

AVAILABLE = all(importlib.util.find_spec(n) is not None for n in ("numpy", "torch"))
if AVAILABLE:
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "course/liftcube/src"))
    from starter.scripted_liftcube import ScriptedLiftCube, SteadyBiasScriptedLiftCube, red_centroid, pixel_to_base, validate_calibration, rotation_step
    from starter.interfaces.action import degrees_to_motor
    from starter.interfaces.observation import Observation


@unittest.skipUnless(AVAILABLE, "Requires course NumPy/Torch (no simulator)")
class ScriptedLiftCubeTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("scipy"), "Packaged joint-chain profiles require SciPy")
    def test_packaged_profiles_select_the_expected_localization(self):
        from contextlib import redirect_stderr, redirect_stdout
        from starter.scripts.evaluate_scripted import main

        configs = Path(__file__).resolve().parents[1] / "course/liftcube/configs/scripted"
        for profile, flags in (("front_rgb", ["--runtime-camera-calibration"]),
                               ("front_rgb", ["--object-scene", __file__, "--runtime-camera-calibration"]),
                               ("privileged_state", ["--privileged-state"]),
                               ("privileged_state", ["--object-scene", __file__, "--privileged-state"]),
                               ("privileged_state", ["--privileged-teacher"])):
            with self.subTest(profile=profile, flags=flags), redirect_stdout(io.StringIO()) as output:
                main(["--calibration", str(configs / (profile + ".json")), "--check", *flags])
                self.assertEqual(json.loads(output.getvalue())["status"], "calibration_checks_passed")
        for profile, flags in (("privileged_state", []), ("front_rgb", ["--privileged-state"]),
                               ("front_rgb", ["--object-scene", __file__, "--privileged-state"]),
                               ("privileged_state", ["--privileged-state", "--runtime-camera-calibration"])):
            with self.subTest(profile=profile, flags=flags), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main(["--calibration", str(configs / (profile + ".json")), "--check", *flags])
                self.assertEqual(error.exception.code, 2)

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "Contour processing requires SciPy")
    def test_contour_recovers_color_and_maintains_inward_target(self):
        from unittest.mock import patch
        from starter.pad_guided import PadGuided, tabletop_component
        from starter.scripted_geometry import GeometricLiftCube

        c=validate_calibration(self.calibration())
        image=np.empty((64,64,3),dtype=np.uint8)
        image[:]=[206,188,161]
        image[38:45,35:41]=[35,65,195]
        mask,color=tabletop_component(image,c)
        self.assertTrue(mask[41,38])
        self.assertEqual(np.count_nonzero(mask),42)
        np.testing.assert_array_equal(color,[35,65,195])
        policy=object.__new__(PadGuided)
        policy.c={'close_motor':0.}
        policy.jaw_blocked=True
        policy.last=np.array([0.,0.,0.,0.,0.,25.])
        with patch.object(GeometricLiftCube,'_step',side_effect=lambda *_: policy.last.copy()):
            for phase,opening,target in [('close',23.,20.),('lift',16.,13.),('hold',2.,0.)]:
                policy.phase=phase
                result=policy._step(image,np.r_[np.zeros(5),opening])
                self.assertEqual(result[5],target)
                np.testing.assert_array_equal(result[:5],policy.last[:5])
                policy.last=result

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "Contour processing requires SciPy")
    def test_partial_occlusion_and_confirmed_target_motion(self):
        from starter.pad_guided import PadGuided, tracked_component
        from starter.scripted_liftcube import GraspStall

        image = np.full((64, 64, 3), [206, 188, 161], dtype=np.uint8)
        color = np.array([40., 75., 180.])
        image[28:36, 27:35] = color
        image[27:37, 31:43] = [245, 210, 24]
        center = np.array([31., 32.])
        visible = tracked_component(image, center, color, 64)
        self.assertEqual(int(visible.sum()), 32)
        self.assertFalse(visible[30, 34])

        controller = object.__new__(PadGuided)
        controller.pixel = center
        controller.occluded = controller.shifted = 0
        controller.close_to_ray = True
        controller.tracked = visible
        controller._check_target(image)
        empty = np.zeros_like(image)
        controller.tracked = None
        controller._check_target(empty)
        shifted = empty.copy()
        shifted[30:34, 48:52, 0] = 255
        for _ in range(3):
            controller._check_target(shifted)
        with self.assertRaises(GraspStall):
            controller._check_target(shifted)

    def calibration(self):
        return dict(schema="so101-front-rgb-rule-v1", evidence="synthetic test only",
            image_shape=[64, 64, 3], K=[[50, 0, 32], [0, 50, 32], [0, 0, 1]],
            camera_to_base=[[1,0,0,0],[0,-1,0,0],[0,0,-1,1],[0,0,0,1]],
            grasp_rotation=np.eye(3).tolist(), joint_sign=[1]*5, joint_offset_deg=[0]*5,
            grasp_xy_offset=[0,0], workspace=[[-.25,-.25,.01],[.25,.25,.3]],
            visible_plane_z=.09, grasp_z=.09, hover_z=.13, lift_z=.17,
            open_motor=100, close_motor=0, joint_names=list("abcde"), tcp_frame="tcp")

    def image(self):
        image = np.zeros((64,64,3), np.uint8)
        image[30:35,30:35,0] = 255
        return image

    def kin(self):
        class Kinematics:
            def forward_kinematics(self, q):
                pose = np.eye(4)
                pose[:3,3] = np.asarray(q[:3])*.001 + [0,0,.15]
                return pose

            def inverse_kinematics(self, q, pose, **kwargs):
                result = q.copy()
                result[:3] = (pose[:3,3]-[0,0,.15])*1000
                return result
        return Kinematics()

    def obs(self, q, image=None):
        # Poison non-permitted fields: a change here must not affect policy output.
        return Observation(self.image() if image is None else image,
            np.r_[q, np.full(15, 999)], metadata={"object_pose": "must never read"})

    def initial(self):
        q = degrees_to_motor(np.array([0,0,30,0,0,0], dtype=float))
        q[5] = 100
        return q

    def test_lost_cube_can_move_across_the_image(self):
        from starter.scripted_geometry import GeometricLiftCube
        policy = object.__new__(GeometricLiftCube)
        policy.c = self.calibration()
        policy.c.update(clear_retry_view=True, camera_to_base=np.asarray(policy.c['camera_to_base']))
        policy.pixel = np.array([14., 32.])
        pose = np.eye(4)
        pose[2, 3] = .18
        policy._pose = lambda q: pose
        image = self.image()
        q = self.initial()
        self.assertFalse(policy._lost_after_lift(image, q))
        self.assertTrue(policy._lost_after_lift(np.roll(image, 18, axis=1), q))
        self.assertFalse(policy._lost_after_lift(np.zeros_like(image), q))
        policy.target = np.ones(3)
        policy.hover_motor = q
        policy._transition('localize')
        self.assertEqual(policy.stable, 0)
        self.assertTrue(all(x is None for x in (policy.pixel, policy.target, policy.hover_motor)))

    def test_detection_and_ambiguity(self):
        np.testing.assert_allclose(red_centroid(self.image()), [32,32])
        for image in (np.zeros((64,64,3), np.uint8), np.zeros((32,32,3), np.uint8)):
            with self.assertRaises(ValueError):
                red_centroid(image)
        image = self.image()
        image[5:8,5:8,0] = 255
        with self.assertRaisesRegex(ValueError, "2 components"):
            red_centroid(image)

    def test_arm_travel_speed_respects_tracking_aperture_and_contact(self):
        p = ScriptedLiftCube(dict(self.calibration(), arm_travel_speed=2.), self.kin())
        q = self.initial()
        p.last = q.copy()
        p.phase = 'hover'
        p.travel_gain = p._travel_gain(q)
        self.assertEqual(p.travel_gain, 2.)
        np.testing.assert_allclose(p._limit_motion(np.array([4,2,0,0,0,8.])), [2,1,0,0,0,1])
        p.last[0] += 7.
        self.assertEqual(p._travel_gain(q), 1.)
        p.last = q.copy()
        for phase in ('descend', 'close', 'lift'):
            p.phase = phase
            self.assertEqual(p._travel_gain(q), 1.)
        p.jaw_blocked = True
        self.assertEqual(p._travel_gain(q), 2.)
        for speed in (True, float('nan'), 2.1):
            with self.assertRaisesRegex(ValueError, 'arm_travel_speed'):
                ScriptedLiftCube(dict(self.calibration(), arm_travel_speed=speed), self.kin())

    def test_open_in_parallel_and_wait_for_measured_aperture(self):
        p = ScriptedLiftCube(dict(self.calibration(), parallel_approach=True), self.kin())
        q = self.initial()
        q[5] = 10
        for _ in range(12):
            sent = p.act(self.obs(q))
            opening = sent[5] > q[5]
            moving = np.max(np.abs(sent[:5]-q[:5])) > .01
            if opening and moving:
                break
            q = sent.copy()
        self.assertTrue(opening and moving)
        self.assertEqual(p.phase, 'hover')
        self.assertFalse(p._jaw_ready(q))
        q[5] = p.c['open_motor']
        self.assertTrue(p._jaw_ready(q))

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "Requires SciPy")
    def test_pad_plan_keeps_lowest_finger_above_table(self):
        from starter.scripted_geometry import GeometricLiftCube
        policy = GeometricLiftCube.__new__(GeometricLiftCube)
        shape = np.array([[-.012, 0., -.103], [.04, -.006, -.105], [.04, .006, -.105]])
        policy.c = dict(visible_plane_z=.015, grasp_z=.012, pad_clearance_vertices_m=shape)
        policy.face_yaw = .25
        policy.retries = 1
        poses = policy._pad_grasps(np.array([.15, -.10, .012]))
        for pose in poses:
            world_height = .012 + (shape - pose['tcp_offset_m']) @ pose['rotation'][2]
            self.assertAlmostEqual(world_height.min(), .0025)
        tilt = [abs(p['rotation'][2, 0]) for p in poses]
        self.assertEqual(tilt, sorted(tilt))

    def test_configured_resolution_keeps_projection_and_detection_contract(self):
        old=validate_calibration(self.calibration())
        for side in (128,256):
            scale=side/64
            c=self.calibration();c['image_shape']=[side,side,3]
            K=np.array(c['K'],float);K[:2,:2]*=scale
            K[:2,2]=(K[:2,2]+.5)*scale-.5;c['K']=K
            c=validate_calibration(c)
            image=np.repeat(np.repeat(self.image(),int(scale),axis=0),int(scale),axis=1)
            pixel=red_centroid(image,c['image_shape'])
            np.testing.assert_allclose(pixel_to_base(pixel,c),pixel_to_base([32,32],old),atol=1e-8)
            with self.assertRaisesRegex(ValueError,'Expected'):
                red_centroid(image)
            image[:]=0;image[5:7,5:7,0]=255
            with self.assertRaisesRegex(ValueError,'0 components'):
                red_centroid(image,c['image_shape'])

    def test_rotation_steps_are_bounded_and_reach_half_turn(self):
        for target in (np.eye(3), np.diag([-1.,-1.,1.])):
            current = np.eye(3)
            for _ in range(160):
                following, remaining = rotation_step(current, target)
                angle = np.arccos(np.clip((np.trace(current.T@following)-1)/2,-1,1))
                self.assertLessEqual(angle, .020001)
                np.testing.assert_allclose(following.T@following, np.eye(3), atol=1e-6)
                current = following
            np.testing.assert_allclose(current,target,atol=1e-6)

    def test_position_arrival_does_not_skip_orientation(self):
        c=self.calibration()
        c['grasp_rotation']=np.diag([-1.,-1.,1.])
        p=ScriptedLiftCube(c,self.kin())
        q=self.initial()
        p.last=q.copy(); p.phase='descend'; p.target=p._pose(q)[:3,3]
        for _ in range(4):
            p.act(self.obs(q))
        self.assertEqual(p.phase,'descend')

    def test_descend_contact_window_requires_lateral_alignment(self):
        for config,dx,phase in [({},0,'descend'),({'relaxed_close_window':True},0,'close'),
                                ({'relaxed_close_window':True},.006,'descend'),
                                ({'face_aligned_grasp':True},0,'close'),
                                ({'face_aligned_grasp':True,'relaxed_close_window':False},0,'descend')]:
            p=ScriptedLiftCube(dict(self.calibration(),**config),self.kin())
            q=self.initial()
            p.last=q.copy()
            p._transition('descend')
            p.target=p._pose(q)[:3,3]+[dx,0,-.004]
            p._solve=lambda xyz,seed,grip,rotation=None:q.copy()
            for _ in range(3):
                p._step(self.image(),q)
            self.assertEqual(p.phase,phase)

    def test_privileged_close_window_requires_planned_face_and_xy_alignment(self):
        from starter.privileged_liftcube import PrivilegedLiftCube
        for tilt, dx, dz, expected in [(20, 0, .007, 'close'), (None, 0, .007, 'descend'),
                                         (20, .006, .007, 'descend'), (20, 0, .009, 'descend')]:
            p = PrivilegedLiftCube(dict(self.calibration(), face_aligned_grasp=True), self.kin())
            q = self.initial()
            p.last = q.copy()
            p._transition('descend')
            p.planned_tilt = tilt
            p.target = p._pose(q)[:3, 3] + [dx, 0, -dz]
            p._solve = lambda xyz, seed, grip, rotation=None: q.copy()
            for _ in range(3):
                p._step(self.image(), q)
            self.assertEqual(p.phase, expected)

    @unittest.skipUnless(importlib.util.find_spec('scipy'), 'Requires SciPy')
    def test_cube_outline_recovers_tabletop_yaw_from_oblique_rgb(self):
        from scipy.spatial import Delaunay
        from starter.scripted_geometry import fit_cube_outline
        c=self.calibration()
        center=np.array([0.,0.,c['visible_plane_z']])
        eye=center+[0.,-.27,.22]
        forward=(center-eye)/np.linalg.norm(center-eye)
        right=np.array([1.,0.,0.])
        camera=np.eye(4)
        camera[:3,:3]=np.column_stack((right,np.cross(forward,right),forward))
        camera[:3,3]=eye
        c.update(camera_to_base=camera,K=[[300,0,128],[0,300,128],[0,0,1]],image_shape=[256,256,3])
        c=validate_calibration(c)
        yaw=.31
        corners=np.array([[x,y,z] for x in (-.015,.015) for y in (-.015,.015) for z in (-.015,.015)])
        corners[:,:2]=corners[:,:2]@[[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]]
        uv=(center+corners-eye)@camera[:3,:3]@c['K'].T
        outline=Delaunay(uv[:,:2]/uv[:,2:])
        y,x=np.indices((256,256))
        image=np.zeros((256,256,3),dtype=np.uint8)
        image[:,:,0]=255*(outline.find_simplex(np.stack((x,y),axis=-1))>=0)
        pixel=red_centroid(image,c['image_shape'])
        point,angle=fit_cube_outline(image,pixel,c)
        np.testing.assert_allclose(point,center,atol=.0015)
        self.assertIsNotNone(angle)
        self.assertLess(abs((angle-yaw+np.pi/4)%(np.pi/2)-np.pi/4),.1)

    def test_slew_preserves_joint_direction(self):
        p=ScriptedLiftCube(self.calibration(),self.kin())
        q=self.initial(); p.last=q.copy()
        requested=q.copy(); requested[:2]+=[4,2]
        p._step=lambda image, measured: requested
        sent=p.act(self.obs(q))
        np.testing.assert_allclose(sent[:2]-q[:2],[1,.5],atol=1e-6)

    def test_reference_does_not_follow_sagging_feedback(self):
        p=ScriptedLiftCube(self.calibration(),self.kin())
        commanded=self.initial()
        measured=commanded.copy(); measured[2]-=5
        p.last=commanded.copy(); p.phase='descend'
        p.target=p._pose(commanded)[:3,3].copy()
        before=p.last.copy()
        p.act(self.obs(measured))
        np.testing.assert_allclose(p.last,before,atol=1e-5)
        self.assertEqual(p.phase,'descend')  # Actual position has not arrived.

    def test_feedback_is_not_validated_as_outgoing_target(self):
        p=ScriptedLiftCube(self.calibration(),self.kin())
        command=self.initial(); command[3]=100
        measured=command.copy(); measured[3]+=.0006
        p.last=command.copy(); p.phase='hold'
        sent=p.act(self.obs(measured))
        np.testing.assert_array_equal(sent,command)
        self.assertEqual(p.phase,'hold')
        self.assertGreater(p.telemetry['measured_motor'][3],100)

    def test_projection_and_invalid_calibration(self):
        c = validate_calibration(self.calibration())
        np.testing.assert_allclose(pixel_to_base([32,32], c), [0,0,.09])
        np.testing.assert_allclose(pixel_to_base([33,32], c), [.0182,0,.09])
        for key, value in (("K", np.zeros((3,3))), ("joint_sign", [0]*5),
                           ("grasp_rotation", np.ones((3,3))), ("lift_z", float("nan"))):
            data = self.calibration(); data[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_calibration(data)
        c["visible_plane_z"] = 2
        with self.assertRaisesRegex(ValueError, "behind"):
            pixel_to_base([32,32], c)

    def test_lift_uses_separate_rotation_with_bounded_interpolation(self):
        c=self.calibration(); c['lift_rotation']=np.diag([-1.,-1.,1.])
        p=ScriptedLiftCube(c,self.kin()); q=self.initial()
        p.last=q.copy(); p.phase='lift'; p.target=p._pose(q)[:3,3].copy()
        captured=[]
        p._solve=lambda xyz, measured, grip, rotation=None: captured.append(rotation.copy()) or measured.copy()
        p.act(self.obs(q))
        angle=np.arccos(np.clip((np.trace(captured[0])-1)/2,-1,1))
        self.assertAlmostEqual(angle,.02,places=6)
        self.assertEqual(p.phase,'lift')

    def test_steady_bias_is_bounded_and_nominal_reference_does_not_drift(self):
        p=SteadyBiasScriptedLiftCube(self.calibration(),self.kin())
        nominal=self.initial(); p.last=nominal.copy(); p.phase='hover'
        p._step=lambda image,q: nominal.copy()
        bias=np.array([0,1.77,1.57,.28,0,0]); q=nominal+bias
        for _ in range(80):
            sent=p.act(self.obs(q)); q=sent+bias
        np.testing.assert_allclose(p.last,nominal,atol=1e-6)
        np.testing.assert_allclose(q[:5],nominal[:5],atol=.001)
        self.assertLessEqual(np.max(np.abs(p.bias)),3)
        p.reset(); self.assertIsNone(p.sent)
        np.testing.assert_array_equal(p.bias,np.zeros(5))

    def test_steady_bias_is_limited_by_remaining_joint_travel(self):
        for direction in (-1, 1):
            with self.subTest(direction=direction):
                p=SteadyBiasScriptedLiftCube(self.calibration(),self.kin())
                nominal=self.initial(); nominal[0]=direction*99
                p.last=nominal.copy(); p.phase='hover'
                p._step=lambda image,q: nominal.copy()
                q=nominal.copy(); q[0]=direction*97
                for _ in range(30):
                    before=p.sent.copy() if p.sent is not None else q.copy()
                    sent=p.act(self.obs(q))
                    self.assertLessEqual(np.max(np.abs(sent-before)),1.0001)
                    self.assertLessEqual(np.max(np.abs(sent[:5])),100)
                self.assertIsNone(p.failure_reason)
                self.assertEqual(p.phase,'hover')
                self.assertAlmostEqual(p.bias[0],direction)
                self.assertAlmostEqual(sent[0],direction*100)
                np.testing.assert_array_equal(p.last,nominal)

    def test_steady_bias_follows_a_changing_joint_margin(self):
        p=SteadyBiasScriptedLiftCube(self.calibration(),self.kin())
        q=self.initial(); q[0]=99
        p.last=q.copy(); p.sent=q.copy(); p.bias[0]=3; p.phase='hover'
        p._step=lambda image,q: q.copy()
        sent=p.act(self.obs(q))
        self.assertIsNone(p.failure_reason)
        self.assertEqual(p.bias[0],1)
        self.assertEqual(sent[0],100)

    def test_contact_close_uses_jaw_feedback_and_keeps_grasp_target(self):
        c=self.calibration();c['contact_close']=True
        p=SteadyBiasScriptedLiftCube(c,self.kin())
        q=self.initial();q[5]=30
        p.last=q.copy();p.phase='close';p.target=np.array([0,0,c['grasp_z']])
        for _ in range(100):
            before=p.sent.copy() if p.sent is not None else q.copy()
            sent=p.act(self.obs(q))
            self.assertLessEqual(np.max(np.abs(sent-before)),1.00001)
            q=sent.copy();q[5]=max(12,q[5])  # Synthetic obstruction, no force API.
            if p.phase=='lift':break
        self.assertEqual(p.phase,'lift',p.failure_reason)
        self.assertTrue(p.jaw_blocked)
        self.assertAlmostEqual(p.grasp_motor,10)
        p._solve=lambda xyz,measured,grip,rotation=None: np.r_[measured[:5],grip]
        self.assertAlmostEqual(p.act(self.obs(q))[5],10)
        p.reset()
        self.assertFalse(p.jaw_blocked)
        self.assertIsNone(p.previous_jaw)
        self.assertEqual(p.grasp_motor,c['close_motor'])

    def test_contact_during_close_recovers_arm_without_reopening(self):
        c=dict(self.calibration(),contact_close=True,hold_contact_on_lift=True,contact_preload_motor=1.)
        p=ScriptedLiftCube(c,self.kin())
        q=self.initial();q[5]=15
        p.last=q.copy();p.last[5]=13
        p.phase='close';p.target=np.array([0.,0.,c['grasp_z']])
        p.contact_stable=2;p.previous_jaw=15.
        sent=p.act(self.obs(q))
        self.assertTrue(p.jaw_blocked)
        self.assertEqual(p.grasp_motor,12.)
        p._retry('wrist tracking lag')
        self.assertTrue(p.resume_loaded)
        self.assertEqual(p.target[2],c['lift_z'])
        self.assertEqual(p._recover(q)[5],12.)
        self.assertEqual(p.phase,'lift')
        p._solve=lambda xyz,seed,grip,rotation=None:np.r_[seed[:5],grip]
        self.assertEqual(p.act(self.obs(q))[5],12.)
        self.assertLessEqual(sent[5],13.)

    def test_loaded_close_probes_upward_before_tracking_guard(self):
        c = dict(self.calibration(), contact_close=True, probe_loaded_close=True)
        p = ScriptedLiftCube(c, self.kin())
        q = self.initial()
        q[5] = 15
        p.last = q.copy()
        p.last[5] = 13
        p.last[3] -= 7
        p.phase = 'close'
        p.target = np.array([0., 0., c['grasp_z']])
        p.act(self.obs(q))
        self.assertEqual(p.phase, 'close')
        q[5] -= .3
        p.act(self.obs(q))
        self.assertEqual(p.phase, 'lift')
        self.assertEqual(p.retries, 0)
        self.assertLess(p.grasp_motor, q[5])

    def test_retry_takes_up_jaw_slack_before_and_during_lift(self):
        c = dict(self.calibration(), contact_close=True, hold_contact_on_lift=True,
                 probe_loaded_close=True, follow_retry_close=True)
        p = ScriptedLiftCube(c, self.kin())
        q = self.initial()
        q[5] = 29
        p.retries, p.phase = 1, "close"
        p.last = q.copy()
        p.last[5] -= 2
        p.last[3] -= 7
        p.target = np.array([0., 0., c["grasp_z"]])
        p.act(self.obs(q))
        p.act(self.obs(q))
        self.assertTrue(p.jaw_blocked)
        self.assertEqual(p.phase, "close")
        q[5] = 27
        p.act(self.obs(q))
        self.assertEqual(p.grasp_motor, p.contact_motor-1.5)
        p._begin_lift()
        p._solve = lambda xyz, seed, grip, rotation=None: np.r_[seed[:5], grip]
        q[5] = 25
        p.act(self.obs(q))
        self.assertEqual(p.grasp_motor, p.contact_motor-1.5)

    def test_slipped_cube_releases_loaded_latch_and_relocalizes(self):
        from starter.privileged_liftcube import PrivilegedLiftCube
        c = dict(self.calibration(), contact_close=True, hold_contact_on_lift=True,
                 retry_dropped_cube=True)
        cube = np.array([0., 0., .09])
        p = PrivilegedLiftCube(c, self.kin(), lambda: cube.copy())
        q = self.initial()
        q[5] = 15
        p.anchor = cube.copy()
        p.pick_offset = cube-p._pose(q)[:3, 3]
        p.grasp_motor = 13.
        p.last = q.copy()
        p.last[5] = 13
        p.target = np.array([0., 0., c['lift_z']])
        p.phase, p.jaw_blocked = 'lift', True
        p._solve = lambda xyz, measured, grip, rotation=None: np.r_[measured[:5], grip]
        cube[0] += .03
        p.act(self.obs(q))
        p.act(self.obs(q))
        self.assertFalse(p.jaw_blocked)
        self.assertEqual(p.phase, 'unload')
        self.assertFalse(p.resume_loaded)
        self.assertTrue(p.open_on_retreat)
        self.assertIsNone(p.failure_reason)

    def test_privileged_replans_from_hover_when_cube_moves(self):
        from starter.privileged_liftcube import PrivilegedLiftCube
        c = dict(self.calibration(), retry_dropped_cube=True, joint_hover_approach=True)
        cube = np.array([0., 0., .09])
        p = PrivilegedLiftCube(c, self.kin(), lambda: cube.copy())
        q = self.initial()
        p.last = q.copy()
        p.anchor = cube.copy()
        p.phase = 'hover'
        p.target = np.array([0., 0., c['hover_z']])
        cube[0] += .025
        p.act(self.obs(q))
        self.assertEqual(p.phase, 'localize')
        for _ in range(2):
            p.act(self.obs(q))
        self.assertEqual(p.phase, 'hover', p.failure_reason)
        self.assertAlmostEqual(p.target[0], cube[0])
        self.assertIsNotNone(p.hover_motor)

    def test_privileged_stall_retracts_before_switching_face_pair(self):
        from starter.privileged_liftcube import PrivilegedLiftCube
        c = dict(self.calibration(), face_aligned_grasp=True)
        p = PrivilegedLiftCube(c, self.kin())
        q = self.initial()
        p.last = p.sent = q.copy()
        p.phase, p.face_pair = "descend", 0
        p.target = np.array([0., 0., c["grasp_z"]])
        pose = np.eye(4)
        pose[2, 3] = c["grasp_z"]
        p._pose = lambda _: pose.copy()
        p._retry("Arm stopped progressing during descend")
        self.assertEqual(p.next_face_pair, 1)
        self.assertEqual(p.phase, "unload")
        p._recover(q)
        self.assertEqual(p.phase, "retreat")
        self.assertEqual(p.retreat_pose[2, 3], c["hover_z"])
        np.testing.assert_equal(p.retreat_pose[:3, :3], pose[:3, :3])
        p.reset()
        self.assertIsNone(p.next_face_pair)

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "Requires SciPy")
    def test_privileged_transport_preserves_height_and_lifts_clear_first(self):
        from starter.privileged_liftcube import PrivilegedLiftCube
        from starter.scripted_liftcube import GraspStall
        c = dict(self.calibration(), face_aligned_grasp=True)
        p = PrivilegedLiftCube(c, self.kin())
        q = self.initial()
        p.last = q.copy()
        xyz = np.array([.08, -.12, c["grasp_z"]])
        def solve(pos, initial, grip, rotation):
            if pos[2] == c["lift_z"] and np.allclose(rotation, np.eye(3)):
                raise GraspStall("Upright waypoint unavailable")
            return initial.copy()
        p._solve = solve
        p._plan_lift(xyz, q)
        np.testing.assert_allclose(p.lift_destination, [*xyz[:2], c["lift_z"]])
        p.target = xyz.copy()
        p._begin_lift()
        np.testing.assert_allclose(p.target, [*xyz[:2], c["hover_z"]])
        self.assertTrue(p.clearing_cube)

    def test_loaded_wrist_lag_allows_other_joints_to_advance(self):
        for cls in (ScriptedLiftCube, SteadyBiasScriptedLiftCube):
            p = cls(dict(self.calibration(), pace_loaded_lift=True), self.kin())
            q = self.initial()
            p.phase, p.jaw_blocked, p.grasp_motor = 'lift', True, 13.
            p.last = q.copy()
            p.last[3] += 7.5
            p.last[5] = p.grasp_motor
            if isinstance(p, SteadyBiasScriptedLiftCube):
                p.sent = p.last.copy()
                p.bias[3] = 1.
            p._step = lambda image, measured: p.last + np.array([0, 1, 0, 1, 0, 0])
            sent = p.act(self.obs(q))
            self.assertEqual(p.phase, 'lift')
            self.assertEqual(p.retries, 0)
            self.assertAlmostEqual(sent[3] - q[3], 7.5)
            self.assertGreater(sent[1], q[1])
            self.assertEqual(sent[5], 13.)

    def test_empty_close_retreats_reopens_and_reacquires_new_image(self):
        c=self.calibration();c['contact_close']=True
        p=ScriptedLiftCube(c,self.kin());q=self.initial();q[5]=30
        p.last=q.copy();p.phase='close';p.target=np.array([0,0,c['grasp_z']])
        for _ in range(100):
            before=q.copy();q=p.act(self.obs(q))
            self.assertLessEqual(np.max(np.abs(q-before)),1.00001)
            if p.phase=='unload':break
        self.assertEqual(p.phase,'unload',p.failure_reason)
        self.assertEqual(p.retries,1)
        observed=set()
        image=np.roll(self.image(),2,axis=1)
        for _ in range(190):
            before=q.copy();q=p.act(self.obs(q,image))
            self.assertLessEqual(np.max(np.abs(q-before)),1.00001)
            observed.add(p.phase)
            if p.phase=='hover':break
        self.assertTrue({'reopen','localize','hover'} <= observed)
        np.testing.assert_allclose(p.pixel,[34,32])
        self.assertAlmostEqual(p.target[0],pixel_to_base(p.pixel,p.c)[0]+.003)
        c['contact_close']='yes'
        with self.assertRaisesRegex(ValueError,'boolean'):
            validate_calibration(c)

    def test_tracking_lag_releases_command_and_preserves_loaded_jaw(self):
        for cls in (ScriptedLiftCube,SteadyBiasScriptedLiftCube):
            with self.subTest(cls=cls):
                p=cls(self.calibration(),self.kin())
                q=self.initial();q[5]=14
                p.last=q.copy();p.last[3]+=8.5
                if isinstance(p,SteadyBiasScriptedLiftCube):
                    p.sent=p.last.copy();p.bias[3]=2
                p.phase='lift';p.target=p._pose(q)[:3,3].copy()
                p._step=lambda image,q:p.last.copy()
                p.act(self.obs(q))
                self.assertEqual(p.phase,'unload')
                before=p.last.copy()
                for _ in range(15):
                    sent=p.act(self.obs(q))
                    self.assertLessEqual(np.max(np.abs(sent-before)),1.00001)
                    self.assertEqual(sent[5],14)
                    before=sent
                    if p.phase=='lift':break
                self.assertEqual(p.phase,'lift')
                self.assertLessEqual(np.max(np.abs(sent[:5]-q[:5])),1)
                self.assertEqual(p.retries,1)
                for _ in range(8):
                    p._retry('persistent jam')
                self.assertEqual(p.phase,'unload')
                self.assertIsNone(p.failure_reason)
                p.reset()
                self.assertEqual(p.retries,0)

    def test_complete_synthetic_sequence_and_reset(self):
        p = ScriptedLiftCube(self.calibration(), self.kin())
        q = self.initial()
        phases = set()
        for _ in range(320):
            old = q.copy()
            image = np.roll(self.image(), -8, axis=0) if p.phase == 'lift' else self.image()
            q = p.act(self.obs(q,image))
            self.assertLessEqual(float(np.max(np.abs(q-old))), 1.0001)
            phases.add(p.phase)
            if p.phase == "hold":
                break
        self.assertEqual(p.phase, "hold", p.failure_reason)
        self.assertTrue({"localize","hover","descend","close","lift","hold"} <= phases)
        self.assertNotIn("success", p.telemetry)
        p.reset()
        self.assertEqual(p.phase, "open")
        self.assertIsNone(p.target)
        self.assertIsNone(p.last)

    def test_missing_cube_is_reacquired_when_visible_again(self):
        p = ScriptedLiftCube(self.calibration(), self.kin())
        q = self.initial()
        for _ in range(4):
            q = p.act(self.obs(q, np.zeros((64,64,3), np.uint8)))
        self.assertEqual(p.phase, "localize")
        self.assertIsNone(p.failure_reason)
        for _ in range(3):
            q = p.act(self.obs(q))
        self.assertEqual(p.phase, "hover")

    def test_privileged_teacher_is_explicit_and_does_not_need_red_pixels(self):
        from starter.privileged_liftcube import PrivilegedLiftCube
        cube = np.array([0., 0., -.01])
        p = PrivilegedLiftCube(self.calibration(), self.kin(), read_cube=lambda: cube.copy())
        q = self.initial()
        blank = np.zeros((64, 64, 3), np.uint8)
        for _ in range(320):
            if p.phase == 'lift':
                cube[2] = p._pose(q)[2, 3]-.10
            q = p.act(self.obs(q, blank))
            if p.phase == 'hold':
                break
        self.assertEqual(p.phase, 'hold', p.failure_reason)
        self.assertTrue(p.telemetry['privileged_teacher'])
        self.assertNotIn('success', p.telemetry)
        p.reset()
        self.assertIsNone(p.anchor)
        missing = PrivilegedLiftCube(self.calibration(), self.kin())
        for _ in range(6):
            missing.act(self.obs(self.initial(), blank))
        self.assertIn('reader', missing.failure_reason)

    def test_cube_visible_on_table_during_lift_triggers_regrasp(self):
        p=ScriptedLiftCube(self.calibration(),self.kin())
        q=self.initial();q[5]=12
        p.last=q.copy();p.pixel=np.array([32,32]);p.target=p._pose(q)[:3,3].copy()
        p._transition('lift')
        p._step=lambda image,q: p.last.copy()
        for tick in (6,9,12):
            p.ticks=tick
            p.act(self.obs(q))
        self.assertEqual(p.phase,'unload')
        self.assertFalse(p.resume_loaded)

    @unittest.skipUnless(importlib.util.find_spec('scipy'), 'Adaptive orientation requires SciPy')
    def test_privileged_orientation_reverts_to_reference_for_next_target(self):
        from starter.privileged_liftcube import PrivilegedLiftCube
        calibration = self.calibration()
        calibration['adaptive_orientation'] = True
        policy = PrivilegedLiftCube(calibration, self.kin())
        reference = policy.c['grasp_rotation'].copy()
        at_limit = True

        def solve(position, measured, grip, rotation=None):
            pose = np.r_[position, 0., 0., grip]
            if at_limit and np.allclose(rotation, reference):
                pose[4] = 99.
            return pose

        def pose(q):
            result = np.eye(4)
            result[:3, 3] = q[:3]
            return result

        policy._solve = solve
        policy._pose = pose
        point = np.array([.06, -.30, .04])
        policy._select_orientations(point)
        self.assertFalse(np.allclose(policy.c['grasp_rotation'], reference))
        policy.reset()
        at_limit = False
        policy._select_orientations(point)
        np.testing.assert_allclose(policy.c['grasp_rotation'], reference)

    def test_rgb_orientation_plans_once_after_stable_image(self):
        c = self.calibration()
        c['adaptive_orientation'] = True
        policy = SteadyBiasScriptedLiftCube(c, self.kin())
        policy.phase = 'localize'
        q = self.initial()
        policy.last = q.copy()
        targets = []
        policy._select_orientations = lambda point: targets.append(point.copy())
        policy._solve = lambda *args, **kwargs: q.copy()
        for _ in range(3):
            policy.act(self.obs(q))
        self.assertEqual(len(targets), 1)
        np.testing.assert_allclose(targets[0][:2], policy.target[:2])
        self.assertEqual(policy.phase, 'hover')

    def test_out_of_range_ik_and_lag_rejected(self):
        p = ScriptedLiftCube(self.calibration(), self.kin())
        with self.assertRaises(ValueError):
            p._solve(np.array([.2,0,.15]), self.initial(), 100)
        p.last = self.initial()
        p.last[0] += 10
        p.phase = "hold"
        p.act(self.obs(self.initial()))
        self.assertEqual(p.phase, "unload")
        self.assertIn("lag", p.retry_reason)

    def test_unreachable_ik_does_not_clip(self):
        kin = self.kin()
        kin.inverse_kinematics = lambda q, pose, **kw: q
        p = ScriptedLiftCube(self.calibration(), kin)
        with self.assertRaisesRegex(ValueError, "residual"):
            p._solve(np.array([.05,0,.15]), self.initial(), 100)

    def test_padding_and_metadata_do_not_influence_actions(self):
        a = ScriptedLiftCube(self.calibration(), self.kin())
        b = ScriptedLiftCube(self.calibration(), self.kin())
        q = self.initial()
        for _ in range(12):
            clean = Observation(self.image(), np.r_[q, np.zeros(15)])
            qa, qb = a.act(clean), b.act(self.obs(q))
            np.testing.assert_array_equal(qa, qb)
            q = qa

    def test_low_initial_pose_refuses_motion(self):
        p = ScriptedLiftCube(self.calibration(), self.kin())
        q = degrees_to_motor(np.array([0,0,-90,0,0,0], dtype=float))
        np.testing.assert_array_equal(p.act(self.obs(q)), q)
        self.assertIn("below", p.failure_reason)

    @unittest.skipUnless(importlib.util.find_spec("yaml"), "Evaluator requires PyYAML")
    def test_evaluation_hold_native_success_and_terminal_recording(self):
        from starter.scripts.evaluate_scripted import RecordedEnv
        from starter.evaluation.evaluate import run_episode
        from starter.interfaces.action import ActionSpec

        q, image = self.initial(), self.image()
        class Env:
            name = "synthetic"
            action_spec = ActionSpec(frequency_hz=60)
            def reset(self, **kwargs):
                self.n = 0
                return Observation(image.copy(), np.r_[q, np.zeros(15)]), {}
            def step(self, action):
                self.n += 1
                terminal_image = image.copy()
                terminal_image[0,0] = self.n
                return Observation(terminal_image, np.r_[action, np.zeros(15)]), 0, self.n == 3, False, {"success": self.n == 3}
        stream = io.StringIO()
        env = RecordedEnv(Env(), stream)
        env.policy = ScriptedLiftCube(self.calibration(), self.kin())
        rec = run_episode(env, env.policy, seed=17, episode_id=0, deterministic=True, checkpoint=None)
        self.assertEqual(rec.length, 3)  # Stop midway through a two-tick hold.
        self.assertTrue(rec.success)  # Only the environment sets success.
        rows = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual(rows[1]["sent_motor"], rows[2]["sent_motor"])
        self.assertEqual(len(env.frames), 4)
        self.assertEqual(env.frames[-1][0,0,0], 3)

    def test_nominal_camera_restored_before_seeded_reset_without_phys_step(self):
        from unittest.mock import MagicMock, call
        from starter.scripts.evaluate_scripted import RecordedEnv

        events, env = MagicMock(), MagicMock()
        camera = events.camera
        env._unwrapped.scene = {"front": camera}
        env._unwrapped.sim = events.sim
        env.reset = events.reset
        env.reset.return_value = (Observation(self.image(), self.initial()), {})
        recorded = RecordedEnv(env, io.StringIO(), nominal_camera=("position", "rotation"))
        recorded.reset(seed=42001)
        self.assertEqual(events.mock_calls, [
            call.camera.set_world_poses("position", "rotation", convention="opengl"),
            call.sim.render(), call.camera.update(0., force_recompute=True), call.reset(seed=42001),
        ])

    def test_calibration_preflight_needs_independent_poses(self):
        from starter.scripts.evaluate_scripted import check_kinematics
        c = self.calibration()
        kin = self.kin()
        c["fk_checks"] = []
        for d in (-10,0,10):
            motor = degrees_to_motor(np.array([d,0,20,0,0,0], float))
            c["fk_checks"].append({"motor": motor.tolist(), "tcp_xyz": [d*.001,0,.17],
                                   "tcp_rotation": np.eye(3).tolist()})
        check_kinematics(ScriptedLiftCube(c, kin))
        c["fk_checks"][0]["tcp_xyz"][0] += .02
        with self.assertRaisesRegex(ValueError, "FK check"):
            check_kinematics(ScriptedLiftCube(c, kin))

    def test_preflight_roundtrip_uses_each_reference_rotation(self):
        from starter.scripts.evaluate_scripted import check_kinematics
        kin=self.kin(); original=kin.forward_kinematics
        def fk(q):
            pose=original(q); a=np.deg2rad(q[0])
            pose[:3,:3]=[[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]]
            return pose
        kin.forward_kinematics=fk
        kin.inverse_kinematics=lambda q,pose,**kw:q.copy()
        c=self.calibration(); c['fk_checks']=[]
        for d in (10,20,30):
            q=np.array([d,0,20,0,0.]); pose=fk(q)
            c['fk_checks'].append(dict(motor=degrees_to_motor(np.r_[q,0]).tolist(),
                tcp_xyz=pose[:3,3].tolist(),tcp_rotation=pose[:3,:3].tolist()))
        check_kinematics(ScriptedLiftCube(c,kin))

    def test_waypoint_preflight_does_not_infer_phase_from_height(self):
        c=self.calibration(); c['hover_z']=c['lift_z']; c['lift_rotation']=np.diag([-1.,-1.,1.])
        p=ScriptedLiftCube(c,self.kin()); q=self.initial()
        p.last=q.copy(); p.phase='localize'; p.stable=2; p.pixel=np.array([32,32])
        rotations=[]
        p._solve=lambda xyz,measured,grip,rotation=None: rotations.append(rotation) or q.copy()
        p.act(self.obs(q))
        np.testing.assert_array_equal(rotations[0],c['grasp_rotation'])
        np.testing.assert_array_equal(rotations[2],c['lift_rotation'])

    def test_geometric_clearance_is_a_height_subgoal_not_success(self):
        if importlib.util.find_spec('scipy') is None:
            self.skipTest('Geometric mode requires SciPy')
        from starter.scripted_geometry import GeometricLiftCube
        c=self.calibration()
        c.update(kinematics_backend='joint_chain',gravity_compensation=True,
            tcp_offset_m=[0,0,0],grasp_asset_sha256='0'*64,
            grasp_presets=[dict(tilt=0,rotation=np.eye(3).tolist(),tcp_offset_m=[0,0,0])])
        kin=self.kin();kin.asset_sha256='0'*64
        p=GeometricLiftCube(c,kin);q=self.initial()
        p.phase='lift';p.clear_lift=True;p.transport_rotation=np.eye(3)
        p.target=np.array([0,0,c['hover_z']]);p.last=q.copy()
        for _ in range(3):
            sent=p.act(self.obs(q))
            self.assertLessEqual(np.max(np.abs(sent-q)),3.00001)
        self.assertFalse(p.clear_lift)
        self.assertEqual(p.phase,'lift')  # No success input or declaration.
        self.assertEqual(p.target[2],c['lift_z'])
        self.assertIsNone(p.failure_reason)
        p.reset()
        self.assertFalse(p.clear_lift)
        self.assertEqual(p.clearance_stable,0)
        c['grasp_asset_sha256']='1'*64
        with self.assertRaisesRegex(ValueError,'same source asset'):GeometricLiftCube(c,kin)
        c['grasp_asset_sha256']='0'*64
        c['grasp_presets'][0]['rotation'][0][0]=2
        with self.assertRaises(ValueError):GeometricLiftCube(c,kin)

    def test_aligned_retry_descends_despite_hover_height_error(self):
        from starter.scripted_geometry import GeometricLiftCube
        c = dict(self.calibration(), kinematics_backend='joint_chain', gravity_compensation=True,
                 tcp_offset_m=[0,0,0], grasp_asset_sha256='0'*64,
                 grasp_presets=[dict(tilt=0, rotation=np.eye(3).tolist(), tcp_offset_m=[0,0,0])])
        kin = self.kin()
        kin.asset_sha256 = c['grasp_asset_sha256']
        p = GeometricLiftCube(c, kin)
        q = self.initial()
        p.last, p.hover_motor = q.copy(), q.copy()
        p.pixel = np.array([32., 32.])
        p.phase, p.retries, p.ticks = 'hover', 4, 210
        pose = p._pose(q)
        p.target = pose[:3, 3] + [0.,0.,.006]
        for _ in range(3):
            p._step(self.image(), q)
        self.assertEqual(p.phase, 'descend')
        self.assertEqual(p.target[2], c['grasp_z'])

    def test_runtime_calibration_uses_camera_base_only_and_records_reset(self):
        if importlib.util.find_spec('scipy') is None:
            self.skipTest('Runtime calibration requires SciPy')
        import torch
        from types import SimpleNamespace
        from scipy.spatial.transform import Rotation
        from starter.camera_calibration import read_camera_to_base
        from starter.scripts.evaluate_scripted import RecordedEnv
        camera=SimpleNamespace(_view=SimpleNamespace(get_world_poses=lambda:(
            torch.tensor([[.4,.6,.8]],dtype=torch.float64),torch.tensor([[1.,0,0,0]],dtype=torch.float64))))
        base_R=Rotation.from_euler('z',90,degrees=True).as_matrix()
        quat=Rotation.from_matrix(base_R).as_quat()[[3,0,1,2]]
        robot=SimpleNamespace(body_names=['base'],data=SimpleNamespace(
            body_pos_w=torch.tensor([[[.1,.2,.3]]],dtype=torch.float64),
            body_quat_w=torch.tensor(quat.reshape(1,1,4),dtype=torch.float64)))
        class Scene:
            def __getitem__(self,key):
                if key=='front':return camera
                if key=='robot':return robot
                raise AssertionError('Non-calibration scene access')
        obs=self.obs(self.initial())
        env=SimpleNamespace(name='synthetic',action_spec=ScriptedLiftCube.action_spec,
            _unwrapped=SimpleNamespace(scene=Scene()),reset=lambda **kwargs:(obs,{}))
        T=read_camera_to_base(env)
        np.testing.assert_allclose(T[:3,:3],base_R.T@np.diag([1.,-1.,-1.]),atol=1e-12)
        np.testing.assert_allclose(T[:3,3],base_R.T@np.array([.3,.4,.5]),atol=1e-12)
        stream=io.StringIO();recorder=RecordedEnv(env,stream,runtime_camera_calibration=True)
        recorder.policy=ScriptedLiftCube(self.calibration(),self.kin())
        recorder.reset(seed=17)
        np.testing.assert_array_equal(recorder.policy.c['camera_to_base'],T)
        self.assertEqual(json.loads(stream.getvalue())['calibration']['source'],'runtime-camera-base-api')

    def test_plan_while_opening_preserves_jaw_motion_and_has_visual_fallback(self):
        from starter.scripted_geometry import GeometricLiftCube
        c=self.calibration()
        c.update(kinematics_backend='joint_chain',gravity_compensation=True,tcp_offset_m=[0,0,0],
            grasp_asset_sha256='0'*64,grasp_presets=[dict(tilt=0,rotation=np.eye(3).tolist(),tcp_offset_m=[0,0,0])])
        def make(enabled):
            config=dict(c,plan_while_opening=enabled);kin=self.kin();kin.asset_sha256='0'*64
            p=GeometricLiftCube(config,kin)
            p.select_pose=lambda point,q:None  # No IK is needed to verify opening.
            return p
        for image,expected in ((self.image(),'hover'),(np.zeros((64,64,3),np.uint8),'localize')):
            old,new=make(False),make(True);q=self.initial();q[5]=0
            for _ in range(120):
                obs=self.obs(q,image);a,b=old.act(obs),new.act(obs)
                np.testing.assert_array_equal(a,b)
                q=a
                if old.phase!='open':break
            self.assertEqual(old.phase,'localize');self.assertEqual(new.phase,expected)
            self.assertIsNone(new.failure_reason)
            if expected=='localize':
                held=new.act(self.obs(q,image))
                self.assertEqual(new.phase,'localize')
                np.testing.assert_array_equal(held,q)
        invalid=dict(c,plan_while_opening='yes')
        with self.assertRaisesRegex(ValueError,'must be boolean'):validate_calibration(invalid)
        invalid=dict(c,plan_while_opening=True,grasp_presets=[])
        with self.assertRaisesRegex(ValueError,'requires geometric'):validate_calibration(invalid)


if __name__ == "__main__":
    unittest.main()
