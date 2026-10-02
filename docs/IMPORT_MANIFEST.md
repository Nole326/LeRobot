# Source import manifest

The original initial commit is preserved; source snapshots are added without rewriting history.

## Included

- Complete pinned LeRobot v0.4.2, LeIsaac 0.4.0 commit and IsaacLab v2.3.0 source trees.
- All tracked code, tests, documentation, hidden configuration, licenses and citations.
- Expanded nested IsaacLab dependency, plus a canonical copy.
- 4,361 upstream file entries including intentional redundancy.
- All 45 LFS test fixtures: 72,385,557 bytes, retrieved from official fixed commits and SHA256-verified.
- 16 independent diagnostic source files and two constraint examples under course/. No course documents or experiment outputs despite that directory name.
- Generic documentation, automated content audit, fixture recovery and manifests.

The sole upstream text change is third_party/lerobot/.gitattributes: seven LFS rules become ordinary binary-file rules. No algorithm/environment/scoring code changed. Original and imported blobs are recorded. Original .gitmodules remains provenance; there are no unresolved root gitlinks.

## External/private boundary

Simulator binaries, CUDA, package distributions and separately licensed scene assets use official distribution channels. Asset URLs/checksums are supplied. Pretrained policies and complete datasets are not fabricated or arbitrarily selected.

Course PDFs, requirement summaries, reports, validation records, screenshots, server inventories/addresses/IDs/reservations and credentials are excluded under the owner's publication scope. Future formal task/evaluation packages must follow their own access rules.

The owner-approved exception is the redacted bilingual project poster and its generic project introduction. This does not authorize publishing the original poster with contact details or other private course materials.

manifests/projects.json records commits/trees; source_files.json validates upstream content; lfs_objects.json records fixtures; course_files.json covers only public utility code and constraints. These files support automated auditing, not private deployment inventory.

Import validation covers source integrity and Python syntax, not fresh GPU deployment, formal evaluation or end-to-end training success.
