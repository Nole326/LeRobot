"""CPU-only source/content checks; stdlib + Git, no upstream code execution."""
import ast
import re
import struct
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r'!?\[[^\]\n]*\]\(([^)\n]+)\)')


def check_markdown(path, root):
    """Check inline relative file links (not remote URLs or anchor targets)."""
    text = path.read_text(encoding='utf-8')
    text = re.sub(r'```.*?```', '', text, flags=re.S)
    errors = []
    for raw in LINK.findall(text):
        target = raw.strip().strip('<>')
        parts = urlsplit(target)
        if parts.scheme or parts.netloc or not parts.path:
            continue
        resolved = (path.parent / unquote(parts.path)).resolve()
        if not resolved.is_relative_to(root.resolve()) or not resolved.exists():
            errors.append(target)
    return errors


def png_size(path):
    with path.open('rb') as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b'\x89PNG\r\n\x1a\n' or header[12:16] != b'IHDR':
        raise ValueError('Not a PNG with an IHDR header')
    return struct.unpack('>II', header[16:24])


def project_files():
    """Include non-ignored new first-party files before staging a change."""
    output = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files', '-z',
                                      '--cached', '--others', '--exclude-standard'])
    names = sorted(set(name for name in output.decode('utf-8').split('\0') if name))
    return [ROOT / name for name in names if not name.startswith('third_party/')]


def main():
    if sys.version_info < (3, 11):
        raise SystemExit('Use Python 3.11+ for the repository checks.')
    subprocess.run([sys.executable, '-B', str(ROOT/'scripts/audit_repository.py')], check=True)
    files = project_files()
    python_count = markdown_count = 0
    for path in files:
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
            python_count += 1
        if path.suffix == '.md':
            errors = check_markdown(path, ROOT)
            if errors:
                raise ValueError(f'Broken/out-of-root local links in {path.relative_to(ROOT)}: {errors}')
            markdown_count += 1
    size = png_size(ROOT/'assets/lerobot-tabletop-pushing-bilingual.png')
    if size != (3840, 2160):
        raise ValueError(f'Unexpected poster dimensions: {size}')
    subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-v'], cwd=ROOT, check=True)
    print(f'PASS: {python_count} project Python files parsed; {markdown_count} Markdown files checked; poster {size}.')
    print('Scope: source/content only; no GPU, network, simulation, training or external-link validation.')


if __name__ == '__main__':
    main()
