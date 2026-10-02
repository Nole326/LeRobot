"""Prevent per-line stretching and inconsistent type roles in the project poster."""
from pathlib import Path
import re
import unittest
import xml.etree.ElementTree as ET

ASSETS = Path(__file__).resolve().parents[1] / 'assets'
NS = '{http://www.w3.org/2000/svg}'


class PosterTypographyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.svg = ET.parse(ASSETS / 'lerobot-tabletop-pushing-bilingual.svg').getroot()
        cls.groups = {g.get('id'): g for g in cls.svg.findall(NS + 'g')
                      if g.get('data-type-role')}

    def test_all_text_has_uniform_role_metadata(self):
        self.assertEqual(len(self.groups), 37)
        roles = {}
        for group in self.groups.values():
            signature = tuple(group.get(k) for k in
                              ('data-font-size-px', 'data-font-weight', 'data-line-height-px'))
            self.assertNotIn(None, signature)
            role = group.get('data-type-role')
            self.assertEqual(roles.setdefault(role, signature), signature)

    def test_glyph_transforms_do_not_stretch(self):
        for group in self.groups.values():
            expected = float(group.get('data-font-size-px')) / 1000
            for path in group.findall(NS + 'path'):
                match = re.fullmatch(r'matrix\(([^)]+)\)', path.get('transform', ''))
                self.assertIsNotNone(match)
                a, b, c, d, _, _ = map(float, match.group(1).split())
                self.assertAlmostEqual(a, expected)
                self.assertAlmostEqual(d, -expected)
                self.assertEqual((b, c), (0, 0))

    def test_matching_section_levels(self):
        for names in [('n1', 'n2', 'n3'), ('h1', 'h2', 'h3'),
                      ('task-cn', 'algorithm-cn', 'deploy-cn'),
                      ('task-en', 'algorithm-en', 'deploy-en'),
                      ('action-cn', 'compare-cn', 'loop-cn', 'interfaces-cn'),
                      ('action-en', 'compare-en', 'loop-en', 'interfaces-en')]:
            roles = {self.groups[name].get('data-type-role') for name in names}
            self.assertEqual(len(roles), 1)
            colors = {p.get('fill') for name in names
                      for p in self.groups[name].findall(NS + 'path')}
            self.assertEqual(len(colors), 1)

    def test_long_sentence_wraps_instead_of_shrinking(self):
        self.assertEqual(len(self.groups['loop-en'].findall(NS + 'path')), 2)
        self.assertEqual(self.groups['loop-en'].get('data-font-size-px'),
                         self.groups['action-en'].get('data-font-size-px'))

    def test_pick_and_place_copy(self):
        expected = {
            'title-cn': '桌面操作',
            'title-right': '/ Pick & Place',
            'task-cn': '多物体视觉抓取与放置',
            'algorithm-cn': '模仿学习 + 强化学习',
            'stable-cn': '抓取后稳定放置',
            'interfaces-cn': '视觉丢失或触发安全条件时暂停，等待人工。',
        }
        for name, text in expected.items():
            self.assertEqual(self.groups[name].get('aria-label'), text)
        copy = ' '.join(g.get('aria-label', '') for g in self.groups.values()).lower()
        for obsolete in ('tabletop pushing', '推动动作', '推入', '2d end-effector', '接触控制'):
            self.assertNotIn(obsolete, copy)

    def test_approved_type_sizes_preserved(self):
        expected = {'number': '72', 'section': '54', 'headline-cn': '68',
                    'headline-en': '58', 'body-cn': '46', 'body-en': '46',
                    'title-cn': '136', 'title-latin': '160'}
        for group in self.groups.values():
            role = group.get('data-type-role')
            if role in expected:
                self.assertEqual(group.get('data-font-size-px'), expected[role])


if __name__ == '__main__':
    unittest.main()
