"""Deployment contracts shared by the published PickOrange policies and their callers."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in ("numpy", "torch", "safetensors")),
    "Residual inference dependencies are optional for the source check",
)
class ResidualModeTests(unittest.TestCase):
    def test_loaders_reject_incompatible_weights_before_loading_tensors(self):
        from course.pickorange.act.residual import EEResidual, make_policy

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            config = folder / "config.json"
            config.write_text(json.dumps({"action_mode": "joint_motor_correction_after_absolute_ee_ik"}))
            with self.assertRaisesRegex(ValueError, "expected action_mode 'absolute_joint_motor'"):
                make_policy(None, {}, folder / "model.safetensors")

            config.write_text(json.dumps({"action_mode": "absolute_joint_motor"}))
            with self.assertRaisesRegex(ValueError, "expected action_mode 'joint_motor_correction"):
                EEResidual(folder)

            config.write_text("{}")
            with self.assertRaisesRegex(ValueError, "got None"):
                make_policy(None, {}, folder / "model.safetensors")


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in ("numpy", "cv2")),
    "EE pose conversion requires NumPy and OpenCV",
)
class ToolFrameTests(unittest.TestCase):
    def test_link_origin_recovers_recorded_tcp_for_two_orientations(self):
        import numpy as np

        from course.pickorange.act.ee_command import link_origin_target

        # Independent reference values from the SO-101 demonstration frame, in metres.
        x, y, z = 0.0423929244, -0.0011997479, -0.0930310568
        origin = np.array([0.2, 0.1, 0.3])
        for yaw, offset in ((0.0, [x, y, z]), (np.pi / 2, [-y, x, z])):
            with self.subTest(yaw=yaw):
                pose = np.r_[origin + offset, 0, 0, yaw, 0.42]
                original = pose.copy()
                result = link_origin_target(pose)
                np.testing.assert_allclose(result[:3], origin, rtol=0, atol=1e-9)
                np.testing.assert_array_equal(result[3:], pose[3:])
                np.testing.assert_array_equal(pose, original)
