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

For LiftCube, follow its [runtime guide](../course/liftcube/README.md): mount this repository at `/repo` and demonstrations separately at `/data/lift_cube`, then run in `/repo/course/liftcube` with the matching image's LeIsaac source on `PYTHONPATH`. The layout below describes the earlier standalone diagnostics, not a requirement to install the new starter over those environments.

The diagnostic launchers expect /workspace/course, application overlays venv_sim and venv_bc, upstream source directories leisaac and lerobot, and assets under /data/course. These are interface path conventions, not a server inventory or authorization to create/restart containers.

In an authorized isolated deployment, map/copy course tools and third_party sources to the expected locations. Check module paths to avoid importing an unrelated installed version. Use a compatible IsaacLab/Isaac Sim runtime; do not overwrite an already running environment simply because this repository contains another source copy.

Simulation and BC application dependencies may conflict in NumPy/packaging versions. The constraint examples are not a universal installation lock or a claim that a fresh installation passes. Review dependency resolution before installing, retain image-shared GPU packages, and freeze the resulting environment privately.

## Validation order

1. Confirm container permissions, resources, device mappings, mounts and cache rules.
2. Verify actual runtime and module paths; obtain assets with checksum validation.
3. Verify reset/step, camera updates, unit conversions, timing, episode boundaries and final observations.
4. For [rule inference](SCRIPTED_LIFTCUBE.md), check independent robot/camera calibration before a short authorized rollout, then freeze the scene set. Demonstrations and ACT optimization are not prerequisites.
5. For learning, separately validate the selected trainer, device, data processing and synchronized checkpoints/RNG; follow the [current LiftCube guide](../course/liftcube/TRAINING.md) rather than automatically running legacy ACT diagnostics. Replay/data acceptance is required when demonstrations are used.
6. Accept the actual task's action SDK, observation restrictions and native scoring before formal evaluation or hardware deployment.

The rule runner dispatches six joint targets. The separate [planar EE development layer](PLANAR_EE_DEVELOPMENT.md) is not an approved replacement for an externally specified controller, and its physical height tracking is not yet accepted. Publishable aggregate development results appear in task guides; raw trajectories, device logs and deployment-specific records stay outside Git. Reproduce results with matching code, calibration and scenario identities, not by combining percentages from different protocols.

## Fixture recovery

All 45 fixtures are already included. To recover a separate verified copy:

```bash
python scripts/fetch_lfs_objects.py --output /approved/asset/directory
```

The tool downloads fixed official commits, verifies size/SHA256, never loads weights or executes assets, and preserves mismatched/partial downloads for diagnosis. Use --limit 1 for a small connectivity test. Destination policies and authorizations still apply.
