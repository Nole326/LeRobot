# Source import manifest

The original initial commit is preserved; source snapshots are added without rewriting history.

## Included

- Complete pinned LeRobot v0.4.2, LeIsaac 0.4.0 commit and IsaacLab v2.3.0 source trees.
- All tracked code, tests, documentation, hidden configuration, licenses and citations.
- Expanded nested IsaacLab dependency, plus a canonical copy.
- 4,361 upstream file entries including intentional redundancy.
- All 45 LFS test fixtures: 72,385,557 bytes, retrieved from official fixed commits and SHA256-verified.
- Standalone diagnostic tools, dependency constraints and the LiftCube starter under course/. Original course documents and experiment outputs are excluded.
- Generic documentation, automated content audit, fixture recovery and manifests.

The sole upstream text change is third_party/lerobot/.gitattributes: seven LFS rules become ordinary binary-file rules. No algorithm/environment/scoring code changed. Original and imported blobs are recorded. Original .gitmodules remains provenance; there are no unresolved root gitlinks.

## External/private boundary

Simulator binaries, CUDA, package distributions and separately licensed scene assets use official distribution channels. Asset URLs/checksums are supplied. Pretrained policies and complete datasets are not fabricated or arbitrarily selected.

Course PDFs, requirement summaries, private reports/full validation records, screenshots, server inventories/addresses/IDs/reservations and credentials are excluded under the owner's publication scope. Task guides may contain approved aggregate development results with their protocol and limitations, not raw run artifacts or deployment details. Future formal task/evaluation packages must follow their own access rules.

The owner-approved exception is the redacted bilingual project poster and its generic project introduction. This does not authorize publishing the original poster with contact details or other private course materials.

manifests/projects.json records commits/trees; source_files.json validates upstream content; lfs_objects.json records fixtures; course_files.json covers public course code, configurations, patches and generic task instructions. These files support automated auditing, not private deployment inventory.

Import validation covers source integrity and Python syntax, not fresh GPU deployment, formal evaluation or end-to-end training success.

## LiftCube starter import

The independent `course/liftcube/` package starts from 45 original source/configuration files. [liftcube_source.json](../manifests/liftcube_source.json) records their archive paths, original/current hashes and modified flags, plus explicitly identified adaptation modules with no original archive entry. The [patch](../course/liftcube/patches/team-interface.patch) reconstructs that managed source set when reversed; it is not an inventory of every later team extension. All public package files, including rule/geometry policies, calibration and planar-control extensions, are covered by [course_files.json](../manifests/course_files.json). New team files are not attributed to the original archive. The original evaluation YAML is unchanged, and repository tests verify source hashes and reverse-patch reconstruction.

This package uses Isaac Sim 4.5.0 / Isaac Lab v2.1.1 / LeIsaac v0.1.2 from the matching image, independently of the newer `third_party/` snapshots. Code, configurations and generic usage instructions are included; simulator archives, datasets, weights, original course instructions and private records are excluded. See the [LiftCube guide](../course/liftcube/README.md).

The image guide records the original SJTU cloud-drive distribution page, archive SHA256 and image ID. A share link does not redistribute the image through Git or change its access/license terms. Data and model identifiers are kept separate from the environment archive.
