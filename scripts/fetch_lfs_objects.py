"""Fetch official fixed-version LFS binaries to a separate directory; no execution."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    target = args.output.resolve()
    if target == ROOT or ROOT in target.parents:
        parser.error('Use a separate dataset/artifact directory outside the source checkout')
    items = json.loads((ROOT / 'manifests/lfs_objects.json').read_text(encoding='utf-8'))
    for item in items[:args.limit]:
        path = (target / item['path']).resolve()
        if not path.is_relative_to(target):
            raise ValueError('Unsafe asset path')
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            with path.open('rb') as existing:
                digest = hashlib.file_digest(existing, 'sha256').hexdigest()
            if path.stat().st_size == item['bytes'] and digest == item['sha256']:
                print('verified', item['path'])
                continue
            raise ValueError('Existing file mismatch; preserved: ' + str(path))
        partial = path.with_name(path.name + '.partial')
        # Exclusive create: never overwrite a failed download or unrelated file.
        with urllib.request.urlopen(urllib.parse.quote(item['url'], safe=':/'), timeout=60) as source, partial.open('xb') as out:
            digest, count = hashlib.sha256(), 0
            while block := source.read(1024 * 1024):
                count += len(block)
                if count > item['bytes']:
                    raise ValueError('Asset exceeds expected size')
                digest.update(block)
                out.write(block)
        if count != item['bytes'] or digest.hexdigest() != item['sha256']:
            raise ValueError('Integrity mismatch; partial retained: ' + str(partial))
        partial.rename(path)
        print('downloaded and verified', item['path'])


if __name__ == '__main__':
    main()
