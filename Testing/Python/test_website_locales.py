"""Keep the static Chinese homepage synchronized with its reviewed source."""

import html
import json
from pathlib import Path
import re
import unittest

from scripts.build_website_locales import ChinesePage, render

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / 'docs/site'


class WebsiteLocaleTests(unittest.TestCase):
    def test_chinese_page_has_complete_reviewed_translations(self):
        expected, missing = render()
        self.assertEqual(missing, [])
        self.assertEqual((SITE / 'zh.html').read_text(encoding='utf-8'), expected)
        self.assertIn('<html lang="zh-CN">', expected)
        for text in ('观看宣传片', '开始使用', '配置需要的依赖环境', '已集成的方法', '参考文献'):
            self.assertIn(text, expected)

    def test_language_links_and_selected_capsule_match_each_page(self):
        for filename, selected in (('index.html', 'en'), ('zh.html', 'zh')):
            source = (SITE / filename).read_text(encoding='utf-8')
            active = re.search(r'<a class="lang-btn is-active"[^>]*>', source).group()
            self.assertIn('data-lang="' + selected + '"', active)
            self.assertIn('aria-current="page"', active)
            self.assertIn('href="index.html"', source)
            self.assertIn('href="zh.html"', source)
            self.assertEqual(source.count('aria-current="page"'), 1)
            self.assertIn('hreflang="zh-CN" href="https://neurodmri.github.io/SlicerDSpace/zh.html"', source)

    def test_translations_keep_images_ids_methods_and_references(self):
        english = (SITE / 'index.html').read_text(encoding='utf-8')
        chinese = (SITE / 'zh.html').read_text(encoding='utf-8')
        for pattern in (r'\bid="([^"]+)"', r'\b(?:src|poster)="([^"]+)"', r'data-method="([^"]+)"'):
            self.assertEqual(re.findall(pattern, english), re.findall(pattern, chinese))
        used = set(re.findall(r'data-reference="(\d+)"', english))
        for citation in json.loads((ROOT / 'docs/website/references.json').read_text(encoding='utf-8'))['references']:
            if str(citation['id']) in used:
                self.assertIn(html.escape(citation['citation']), chinese)
                self.assertIn(html.escape(citation['url']), chinese)

    def test_shell_commands_are_literal_but_agent_prompts_are_translated(self):
        english = (SITE / 'index.html').read_text(encoding='utf-8')
        chinese = (SITE / 'zh.html').read_text(encoding='utf-8')
        self.assertEqual(re.findall(r'<pre[^>]*><code>(.*?)</code></pre>', english, re.S),
                         re.findall(r'<pre[^>]*><code>(.*?)</code></pre>', chinese, re.S))
        prompts = re.findall(r'<code id="agent-[^"]+">(.*?)</code>', chinese, re.S)
        self.assertEqual(len(prompts), 2)
        self.assertTrue(all('SKILL.md' in text and '参考' in text for text in prompts))

    def test_unknown_new_copy_is_not_silently_left_english(self):
        parser = ChinesePage({})
        parser.feed('<p>New installation instructions.</p>')
        self.assertIn('New installation instructions.', parser.missing)

    def test_canonical_url_uses_the_localized_page(self):
        chinese = (SITE / 'zh.html').read_text(encoding='utf-8')
        self.assertIn('rel="canonical" href="https://neurodmri.github.io/SlicerDSpace/zh.html"', chinese)

    def test_website_generator_is_not_shipped_in_the_extension_runtime(self):
        from scripts.extension_runtime.runtime_resources import runtime_files
        self.assertNotIn('scripts/build_website_locales.py', runtime_files(ROOT))


if __name__ == '__main__':
    unittest.main()
