# Third-party notices

Original license, copyright, citation and per-file notices are retained. This integration is not endorsed by upstream authors.

| Project | Commit | License |
| --- | --- | --- |
| Hugging Face LeRobot | `58f70b6bd370864139a3795ac3497a9eae8c42d5` | `third_party/lerobot/LICENSE`: Apache-2.0 and incorporated-component notices |
| Lightwheel LeIsaac | `24d3bcd3f1e4585740fc79921782c41617237812` | `third_party/leisaac/LICENSE`: Apache-2.0 |
| Isaac Lab | `3c6e67bb5c7ada942a6d1884ab69338f57596f77` | `third_party/isaaclab/LICENSE`: BSD-3-Clause; also LICENSE-mimic and per-file notices |

LeIsaac's nested IsaacLab dependency is expanded, retaining all notices. Its original .gitmodules remains as provenance, not as a root submodule configuration.

All 45 pinned LeRobot LFS test fixtures were retrieved from official fixed-commit URLs and SHA256-verified. They are stored as ordinary Git binaries, not presented as pretrained application policies. The only modified upstream text is LeRobot .gitattributes, disabling LFS filters for these materialized files. Original pointer hashes and binary digests remain recorded.

Isaac Sim, NVIDIA images, CUDA, package distributions, pretrained weights and separately released simulation assets have their own terms; obtain them from official sources under those terms. They are not relicensed here.

Private course and server/experiment records are excluded. No blanket new license is imposed on original integration material during this import; the owner may choose one separately. Upstream code remains under its original terms.
