"""Source packaging checks; do not import Torch or start the simulator."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LiftCubeImportTests(unittest.TestCase):
    def test_source_identity_and_unchanged_evaluation_config(self):
        manifest = json.loads((ROOT / "manifests/liftcube_source.json").read_text(encoding="utf-8"))
        records = manifest["files"]
        self.assertGreaterEqual(len(records), 46)
        self.assertEqual(sum(row["archive_path"] is not None for row in records), 45)
        self.assertEqual(len({row["path"] for row in records}), len(records))
        for row in records:
            with self.subTest(path=row["path"]):
                self.assertEqual(hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest(), row["sha256"])
        evaluation = next(row for row in records if row["path"].endswith("configs/eval_lift_cube.yaml"))
        self.assertFalse(evaluation["modified"])
        self.assertEqual(evaluation["original_sha256"], evaluation["sha256"])

    def test_patch_reconstructs_original_source(self):
        records = json.loads((ROOT / "manifests/liftcube_source.json").read_text(encoding="utf-8"))["files"]
        patch = ROOT / "course/liftcube/patches/team-interface.patch"
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory)
            for row in records:
                target = scratch / Path(row["path"]).relative_to("course/liftcube")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / row["path"], target)
            subprocess.run(["git", "-c", "core.autocrlf=false", "apply", "--reverse", "--whitespace=nowarn", str(patch)],
                           cwd=scratch, check=True, capture_output=True)
            for row in records:
                target = scratch / Path(row["path"]).relative_to("course/liftcube")
                with self.subTest(path=row["path"]):
                    if row["original_sha256"] is None:
                        self.assertFalse(target.exists())
                    else:
                        self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), row["original_sha256"])


if __name__ == "__main__":
    unittest.main()
