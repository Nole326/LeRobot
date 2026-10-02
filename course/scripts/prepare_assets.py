"""Fetch only the official public PickOrange example, verify SHA, extract safely."""
import hashlib
import json
from pathlib import Path
import urllib.request
import argparse
import shutil
import zipfile

root = Path('/data/course/assets')
downloads = Path('/data/course/downloads')
root.mkdir(parents=True, exist_ok=True)
downloads.mkdir(parents=True, exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument('--local-source', type=Path)
args = parser.parse_args()
# Sizes/digests recorded from the official v0.1.0 release API before transfer.
release = {'assets': [
    {'name': 'kitchen_with_orange.zip', 'size': 72918086, 'browser_download_url': 'https://github.com/LightwheelAI/leisaac/releases/download/v0.1.0/kitchen_with_orange.zip'},
    {'name': 'so101_follower.usd', 'size': 23241268, 'browser_download_url': 'https://github.com/LightwheelAI/leisaac/releases/download/v0.1.0/so101_follower.usd'},
]}
expected = {
    'kitchen_with_orange.zip': 'd314c54b63a17e91402bfaddf26e21ff614adf2430fa092b78897f15b8adea34',
    'so101_follower.usd': '64a877c3b82cdc4a48ab8a1f321a2dd3ef7c55d4b10bce222b58c530d978ae58',
}
manifest = []
for asset in release['assets']:
    name = asset['name']
    if name not in expected:
        continue
    dest = downloads / name
    if args.local_source:
        source = args.local_source / name
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected[name]
        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() != expected[name]:
            dest.rename(dest.with_suffix(dest.suffix + '.partial-evidence'))
        shutil.copy2(source, dest)
    if not dest.exists():
        print('Downloading', name, asset['size'], flush=True)
        with urllib.request.urlopen(asset['browser_download_url'], timeout=120) as stream, dest.open('wb') as output:
            while block := stream.read(1024 * 1024):
                output.write(block)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    if digest != expected[name] or dest.stat().st_size != asset['size']:
        raise RuntimeError(f'Asset integrity mismatch: {name}; retain file for diagnosis')
    if name.endswith('.zip'):
        scene_root = root / 'scenes'
        scene_root.mkdir(exist_ok=True)
        with zipfile.ZipFile(dest) as archive:
            for member in archive.infolist():
                target = (scene_root / member.filename).resolve()
                if not target.is_relative_to(scene_root.resolve()):
                    raise RuntimeError(f'Unsafe archive member: {member.filename}')
            archive.extractall(scene_root)
    else:
        (root / 'robots').mkdir(exist_ok=True)
        shutil.copy2(dest, root / 'robots' / name)
    manifest.append({'name': name, 'bytes': dest.stat().st_size, 'sha256': digest, 'source': asset['browser_download_url']})
assert (root / 'scenes/kitchen_with_orange/scene.usd').is_file()
(root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2), flush=True)
