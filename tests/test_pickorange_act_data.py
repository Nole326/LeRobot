"""Check the two archive layouts, label timing and skill-boundary padding."""

import json
import tempfile
import unittest
import zlib
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "NumPy is optional for source checks")
class TestPickOrangeAct(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib.util

        path = Path(__file__).resolve().parents[1] / "course/pickorange/act/data.py"
        spec = importlib.util.spec_from_file_location("pickorange_act_data", path)
        cls.data = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.data)

    def dataset(self, root, mode):
        folder = root / "episodes/episode-000000"
        folder.mkdir(parents=True)
        count = 10
        joints = np.tile(np.arange(count)[:, None], (1, 6)).astype(np.float32)
        actions = np.zeros((count, 7 if mode == "ee" else 6), dtype=np.float32)
        actions[:, 0] = np.arange(count)
        np.savez(
            folder / "demonstration.npz",
            joints=joints,
            action=actions,
            timestamp=np.arange(count) / 60,
        )
        names = ("demonstration.npz", "front.mkv", "wrist.mkv")
        path = folder.relative_to(root)
        entry = {
            "seed": 10,
            "records": count,
            "path": str(path),
            "segments": [[0, 4], [4, 7], [7, 10]],
        }
        if mode == "ee":
            entry["files"] = dict.fromkeys(names, 1)
        else:
            entry.pop("path")
            entry["files"] = [{"path": str(path / name)} for name in names]
        manifest = root / "dataset.json"
        manifest.write_text(
            json.dumps(
                {
                    "episodes": [entry],
                    "control": "absolute_ee_robot_base",
                    "physics_hz": 60,
                    "action_hz": 30,
                }
            ),
            encoding="utf-8",
        )
        data = self.data.Episodes(manifest, mode)
        frame = zlib.compress(bytes(480 * 640 * 3))
        data.views = [([frame] * count, [frame] * count)]
        return data

    def test_stride_two_targets(self):
        with tempfile.TemporaryDirectory() as td:
            data = self.dataset(Path(td), "qpos")
            _, a, pad, *_ = data.sample((0, 0, 10))
            targets = a * data.astd + data.amu
            np.testing.assert_allclose(targets[:4, 0], [2, 4, 6, 8])
            self.assertEqual(pad.tolist(), [False] * 4 + [True] * 11)

    def test_segment_boundary_and_encoding(self):
        with tempfile.TemporaryDirectory() as td:
            data = self.dataset(Path(td), "ee")
            _, a, pad, *_ = data.sample((0, 1, 4))
            targets = a * data.astd + data.amu
            np.testing.assert_allclose(targets[:4, 0], [1, 2, 3, 3], atol=1e-6)
            np.testing.assert_array_equal(targets[:4, -1], [0, 1, 1, 1])
            np.testing.assert_array_equal(targets[0, 3:9], [1, 0, 0, 0, 1, 0])
            self.assertEqual(pad.tolist(), [False] * 3 + [True] * 12)

    def test_rejects_30_hz_qpos(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.dataset(root, "qpos")
            path = root / "episodes/episode-000000/demonstration.npz"
            with np.load(path) as arrays:
                joints, actions = arrays["joints"], arrays["action"]
            np.savez(path, joints=joints, action=actions, timestamp=np.arange(len(joints)) / 30)
            with self.assertRaisesRegex(ValueError, "60 Hz"):
                self.data.Episodes(root / "dataset.json", "qpos")

    def test_rejects_delta_ee_labels(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.dataset(root, "ee")
            manifest = root / "dataset.json"
            spec = json.loads(manifest.read_bytes())
            spec["control"] = "delta_ee"
            manifest.write_text(json.dumps(spec), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "absolute EE"):
                self.data.Episodes(manifest, "ee")
