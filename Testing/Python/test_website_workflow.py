"""The website workflow keeps the approved figure's geometry and display data."""

from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]


class WebsiteWorkflowTests(unittest.TestCase):
    def test_dark_derivative_preserves_geometry_and_images(self):
        import sys
        sys.path.insert(0, str(ROOT / 'scripts'))
        from prepare_website_workflow import render
        source = (ROOT / 'docs/site/assets/img/workflow-overview.svg').read_bytes()
        before, after = ET.fromstring(source), ET.fromstring(render(source))
        frames = [n for n in after.iter() if n.get('data-workflow-frame')]
        self.assertEqual(len(frames), 10)
        self.assertTrue(all(n.get('stroke') == '#ffffff' and n.get('fill') == 'none' for n in frames))
        parents = {child: parent for parent in after.iter() for child in parent}
        rounded = [n for n in frames if n.get('rx') == '18']
        self.assertEqual(len(rounded), 7)
        for frame in rounded:
            self.assertFalse(parents[frame].get('clip-path'))
            self.assertEqual(frame.get('vector-effect'), 'non-scaling-stroke')
            self.assertEqual(frame.get('stroke-width'), '1')
        for tag in ('text', 'image'):
            a, b = list(before.iter('{http://www.w3.org/2000/svg}' + tag)), list(after.iter('{http://www.w3.org/2000/svg}' + tag))
            self.assertEqual(len(a), len(b))
            for original, output in zip(a, b):
                self.assertEqual(original.text, output.text)
                for attribute in ('x', 'y', 'width', 'height', 'href'):
                    self.assertEqual(original.get(attribute), output.get(attribute))

    def test_both_locales_use_the_new_vector_figure(self):
        for name in ('index.html', 'zh.html'):
            html = (ROOT / 'docs/site' / name).read_text(encoding='utf-8')
            section = html.split('id="choose"', 1)[1].split('</section>', 1)[0]
            self.assertIn('assets/img/workflow-overview.svg', section)
            self.assertNotIn('fig-product-map', section)

    def test_footer_uses_font_free_lab_wordmarks_and_lab_attribution(self):
        ns = '{http://www.w3.org/2000/svg}'
        outlines = []
        for name in ('fans-lab-wordmark.svg', 'fans-lab-wordmark-light.svg'):
            root = ET.parse(ROOT / 'docs/site/assets/img' / name).getroot()
            self.assertFalse(list(root.iter(ns + 'image')))
            self.assertFalse(list(root.iter(ns + 'text')))
            self.assertFalse(list(root.iter(ns + 'rect')))
            paths = [node.get('d') for node in root.iter(ns + 'path')]
            self.assertGreater(len(paths), 5)
            outlines.append(paths)
        self.assertEqual(*outlines)
        for name in ('index.html', 'zh.html'):
            footer = (ROOT / 'docs/site' / name).read_text(encoding='utf-8').split('<footer', 1)[1]
            self.assertNotIn('fans-lab-wordmark', footer)
            self.assertIn("© 2026 Chengjin Li from Fan's Lab.", footer)


if __name__ == '__main__':
    unittest.main()
