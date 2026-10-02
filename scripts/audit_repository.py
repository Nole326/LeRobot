"""Audit vendored Git source blobs without importing or executing upstream code.

--write-manifests is a maintainer-only import operation requiring refs/vendor/*.
The default works after an ordinary clone and makes no changes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = [
    ('lerobot', 'huggingface/lerobot', '58f70b6bd370864139a3795ac3497a9eae8c42d5', 'v0.4.2'),
    ('leisaac', 'LightwheelAI/leisaac', '24d3bcd3f1e4585740fc79921782c41617237812', '0.4.0'),
    ('isaaclab', 'isaac-sim/IsaacLab', '3c6e67bb5c7ada942a6d1884ab69338f57596f77', 'v2.3.0'),
]
LFS = re.compile(rb'\Aversion https://git-lfs.github.com/spec/v1\r?\noid sha256:([0-9a-f]{64})\r?\nsize ([0-9]+)')


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def blob_hash(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def content(path):
    return path.readlink().as_posix().encode() if path.is_symlink() else path.read_bytes()


def write_manifests():
    records, projects = [], []
    previous_lfs_path = ROOT / 'manifests/lfs_objects.json'
    previous_lfs = {row['path']: row for row in json.loads(previous_lfs_path.read_text(encoding='utf-8'))} if previous_lfs_path.exists() else {}
    for name, upstream, commit, version in PROJECTS:
        actual = git('rev-parse', 'refs/vendor/' + name).decode().strip()
        if actual != commit:
            raise ValueError(f'Wrong source reference: {name}')
        tree = git('rev-parse', commit + '^{tree}').decode().strip()
        projects.append(dict(name=name, url='https://github.com/' + upstream,
                             commit=commit, version=version, tree=tree))
        for entry in git('ls-tree', '-rz', commit).split(b'\0'):
            if not entry:
                continue
            meta, raw_path = entry.split(b'\t', 1)
            mode, kind, sha = meta.decode().split()
            path = raw_path.decode()
            if kind == 'commit':
                if not (name == 'leisaac' and path == 'dependencies/IsaacLab'
                        and sha == PROJECTS[2][2]):
                    raise ValueError('Unaccounted nested repository: ' + path)
                continue  # Fully expanded and validated below.
            relative = 'third_party/' + name + '/' + path
            records.append(dict(path=relative, blob=sha, mode=mode, upstream=upstream,
                                commit=commit, upstream_path=path))
    for record in list(records):
        if record['path'].startswith('third_party/isaaclab/'):
            clone = dict(record)
            clone['path'] = record['path'].replace('third_party/isaaclab/',
                                                  'third_party/leisaac/dependencies/IsaacLab/', 1)
            records.append(clone)
    records.sort(key=lambda row: row['path'])
    lfs = []
    for record in records:
        data = content(ROOT / record['path'])
        if blob_hash(data) != record['blob']:
            if record['path'] == 'third_party/lerobot/.gitattributes':
                original = git('cat-file', 'blob', record['blob'])
                expected = original.replace(b'filter=lfs diff=lfs merge=lfs -text', b'-filter -diff -merge -text')
                if data != expected:
                    raise ValueError('Unexpected LFS integration change')
                record['imported_blob'] = blob_hash(data)
                record['adjustment'] = 'Store verified LFS test binaries in ordinary Git; no model/source algorithm changes'
            elif record['path'] in previous_lfs:
                item = previous_lfs[record['path']]
                if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
                    raise ValueError('Hydrated LFS mismatch: ' + record['path'])
                record['hydrated_sha256'] = item['sha256']
                record['hydrated_bytes'] = item['bytes']
                lfs.append(item)
            else:
                raise ValueError('Source mismatch: ' + record['path'])
        match = LFS.match(data)
        if match:
            lfs.append(dict(path=record['path'], sha256=match[1].decode(), bytes=int(match[2]),
                            url=f"https://media.githubusercontent.com/media/{record['upstream']}/{record['commit']}/{record['upstream_path']}"))
    course = []
    for path in sorted((ROOT / 'course').rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
            data = path.read_bytes()
            course.append(dict(path=path.relative_to(ROOT).as_posix(), bytes=len(data),
                               sha256=hashlib.sha256(data).hexdigest()))
    manifests = ROOT / 'manifests'
    manifests.mkdir(exist_ok=True)
    for name, value in [('source_files', records), ('projects', projects),
                        ('lfs_objects', lfs), ('course_files', course)]:
        (manifests / (name + '.json')).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')


def audit():
    records = json.loads((ROOT / 'manifests/source_files.json').read_text(encoding='utf-8'))
    expected_paths = {record['path'] for record in records}
    actual_paths = {p.relative_to(ROOT).as_posix() for p in (ROOT / 'third_party').rglob('*')
                    if p.is_file() or p.is_symlink()}
    if expected_paths != actual_paths:
        raise ValueError(f'File-set mismatch: missing={sorted(expected_paths-actual_paths)[:5]}, extra={sorted(actual_paths-expected_paths)[:5]}')
    for record in records:
        data = content(ROOT / record['path'])
        if 'hydrated_sha256' in record:
            if len(data) != record['hydrated_bytes'] or hashlib.sha256(data).hexdigest() != record['hydrated_sha256']:
                raise ValueError('LFS binary mismatch: ' + record['path'])
        elif blob_hash(data) != record.get('imported_blob', record['blob']):
            raise ValueError('Upstream blob mismatch: ' + record['path'])
    course = json.loads((ROOT / 'manifests/course_files.json').read_text(encoding='utf-8'))
    for record in course:
        if hashlib.sha256((ROOT / record['path']).read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Course file mismatch: ' + record['path'])
    forbidden = re.compile(r'(^|/)(\.netrc|\.env|secrets|personal_config\.md|\.local-downloads)(/|$)|\.(pem|key)$|^course/(docs/|validation_|README\.md)')
    staged = git('ls-files', '-z').decode().split('\0')
    tracked_course = {path for path in staged if path.startswith('course/')}
    if tracked_course != {row['path'] for row in course}:
        raise ValueError('Public utility manifest differs from tracked files')
    if any(forbidden.search(path) for path in staged if path):
        raise ValueError('Forbidden private file is tracked')
    lfs = json.loads((ROOT / 'manifests/lfs_objects.json').read_text(encoding='utf-8'))
    print(json.dumps(dict(status='passed', upstream_files=len(records), course_files=len(course),
                          lfs_objects=len(lfs), lfs_hydrated=sum('hydrated_sha256' in x for x in records), lfs_bytes=sum(x['bytes'] for x in lfs),
                          scope='source/content only; no simulation, installation or training'), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--write-manifests', action='store_true')
    args = parser.parse_args()
    if args.write_manifests:
        write_manifests()
    audit()
