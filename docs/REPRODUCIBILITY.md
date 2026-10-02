# Getting and using the source

```bash
git clone -c core.longpaths=true -c core.autocrlf=false https://github.com/Nole326/LeRobot.git
cd LeRobot
python scripts/check_project.py
```

No submodule initialization or LFS download is needed. Manifests preserve baseline commits/trees, original blob hashes and materialized fixture SHA256 values. Auditing checks files without executing upstream code.

Use Git and Python 3.11+ for these CPU-only checks. The unified entry point also parses project Python files, checks local Markdown file links and poster dimensions, and runs checker unit tests. It does not test GPU deployment. See [BASELINE.md](BASELINE.md) to check out the fixed initial version.

The clone-local settings support IsaacLab's deep paths on Windows and preserve baseline line endings. They do not change global Git or OS settings. A long Windows destination path may fail to check out without core.longpaths even though the remote objects are complete.

## Generic deployment layout

The diagnostic launchers expect /workspace/course, application overlays venv_sim and venv_bc, upstream source directories leisaac and lerobot, and assets under /data/course. These are interface path conventions, not a server inventory or authorization to create/restart containers.

In an authorized isolated deployment, map/copy course tools and third_party sources to the expected locations. Check module paths to avoid importing an unrelated installed version. Use a compatible IsaacLab/Isaac Sim runtime; do not overwrite an already running environment simply because this repository contains another source copy.

Simulation and BC application dependencies may conflict in NumPy/packaging versions. The constraint examples are not a universal installation lock or a claim that a fresh installation passes. Review dependency resolution before installing, retain image-shared GPU packages, and freeze the resulting environment privately.

## Validation order

1. Confirm container permissions, resources, device mappings, mounts and cache rules.
2. Verify actual runtime and module paths; obtain assets with checksum validation.
3. Test CUDA, synthetic ACT updates, dataset read/write and unit conversions without real robot motion.
4. Verify reset/step, camera updates, episode boundaries, final observations, action replay and data conversion.
5. Validate the actual task's controller/units, observation/scoring constraints and synchronized training/RNG checkpoints before training.

The EE-control and rollout scripts include diagnostic experiments, not approved replacements for an externally specified controller. Private validation results are deliberately not published.

## Fixture recovery

All 45 fixtures are already included. To recover a separate verified copy:

```bash
python scripts/fetch_lfs_objects.py --output /approved/asset/directory
```

The tool downloads fixed official commits, verifies size/SHA256, never loads weights or executes assets, and preserves mismatched/partial downloads for diagnosis. Use --limit 1 for a small connectivity test. Destination policies and authorizations still apply.
