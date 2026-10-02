# Dependency inventory

## Complete source snapshots

| Project | Fixed official source | Included |
| --- | --- | --- |
| LeRobot | https://github.com/huggingface/lerobot/tree/58f70b6bd370864139a3795ac3497a9eae8c42d5 | Entire tracked tree: policies, training, data, hardware, tests, docs, package metadata |
| LeIsaac | https://github.com/LightwheelAI/leisaac/tree/24d3bcd3f1e4585740fc79921782c41617237812 | Entire tracked tree, with expanded IsaacLab dependency |
| IsaacLab | https://github.com/isaac-sim/IsaacLab/tree/3c6e67bb5c7ada942a6d1884ab69338f57596f77 | Entire framework, controllers, tasks, tests, docs and build files |
| Integration utilities | course/scripts and scripts | Independent diagnostic tools, constraint examples, source audit and fixture recovery |

ACT, SmolVLA and SAC/HIL-SERL implementations are already in LeRobot; separate repositories are not needed for those included implementations. Optional backends/hardware extras remain declared in original package metadata; this is not a claim they are all installed.

## External components

| Component | Source | Treatment |
| --- | --- | --- |
| Isaac Sim / runtime | nvcr.io/nvidia/isaac-lab:2.3.0 | Obtain official image under its terms; verify actual runtime after pull. No image binaries in Git |
| Python dependencies | Upstream pyproject/extension metadata; course/constraints-*.txt | Resolve in compatible isolated application environments, retaining image-shared torch/CUDA |
| PickOrange kitchen / SO101 USD | manifests/external_assets.json and course/scripts/prepare_assets.py | Official URLs, sizes and SHA256; download to approved asset storage |
| LFS test fixtures | manifests/lfs_objects.json | All 45 binaries are included and verified; recovery tool supplied |
| Pretrained policies | Upstream model docs/cards | Select version and record digest before use; no arbitrary weights claimed as included |
| Real-robot calibration | LeRobot hardware modules and optional dependencies | Configure for the actual device; private calibrations not published |
| Formal task environment/scoring | External task provider | Supplied separately; no invented implementation of undisclosed protocols |

This is a complete snapshot of the three selected application projects, not an offline mirror of every Python/CUDA/system package. Additional repositories become dependencies when actually selected and pinned. RoboTwin/rl-garden are not runtime dependencies here.
