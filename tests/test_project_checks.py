import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from check_project import check_markdown, png_size, project_files


class ProjectChecksTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.doc = self.root/'README.md'

    def test_query_and_encoded_path(self):
        (self.root/'poster image.png').touch()
        self.doc.write_text('[image](poster%20image.png?v=2)', encoding='utf-8')
        self.assertEqual(check_markdown(self.doc, self.root), [])

    def test_missing_file(self):
        self.doc.write_text('[missing](missing.md)', encoding='utf-8')
        self.assertEqual(check_markdown(self.doc, self.root), ['missing.md'])

    def test_parent_escape(self):
        self.doc.write_text('[outside](../outside.md)', encoding='utf-8')
        self.assertEqual(check_markdown(self.doc, self.root), ['../outside.md'])

    def test_remote_and_anchor_ignored(self):
        self.doc.write_text('[web](https://example.org/a) [section](#example)', encoding='utf-8')
        self.assertEqual(check_markdown(self.doc, self.root), [])

    def test_code_examples_ignored(self):
        self.doc.write_text('```md\n[example](missing.md)\n```', encoding='utf-8')
        self.assertEqual(check_markdown(self.doc, self.root), [])

    def test_png_dimensions(self):
        p = self.root/'sample.png'
        p.write_bytes(b'\x89PNG\r\n\x1a\n'+struct.pack('>I',13)+b'IHDR'+struct.pack('>II',3840,2160))
        self.assertEqual(png_size(p), (3840,2160))

    def test_reject_non_png(self):
        p = self.root/'bad.png'
        p.write_bytes(b'not a png')
        with self.assertRaises(ValueError):
            png_size(p)

    def test_new_files_checked_but_ignored_files_excluded(self):
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        (self.root/'.gitignore').write_text('private/\n', encoding='utf-8')
        (self.root/'new.py').write_text('pass\n', encoding='utf-8')
        (self.root/'private').mkdir()
        (self.root/'private/secret.md').write_text('not inspected', encoding='utf-8')
        (self.root/'third_party').mkdir()
        (self.root/'third_party/vendor.py').touch()
        with patch('check_project.ROOT', self.root):
            names = {p.relative_to(self.root).as_posix() for p in project_files()}
        self.assertEqual(names, {'.gitignore', 'new.py'})


if __name__ == '__main__':
    unittest.main()
