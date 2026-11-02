"""Keep the homepage aligned with the published assets and current GUI entry points."""
import html
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
TITLE = "SlicerDSpace"


class WebsiteEditorialCopyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / "docs/site/index.html").read_text(encoding="utf-8")

    def section(self, name):
        return re.search(r'<section\b[^>]*\bid="' + name + r'"[^>]*>(.*?)</section>',
                         self.source, re.S).group(1)

    def test_hero_is_wordmark_and_two_film_setup_links(self):
        hero = re.search(r"<h1\b[^>]*>(.*?)</h1>", self.source, re.S).group(1)
        self.assertEqual(html.unescape(re.sub(r"<[^>]+>", "", hero)).strip(), TITLE)
        self.assertIn('src="assets/img/slicerdspace-wordmark-light.svg"', hero)
        self.assertIn("<title>" + TITLE + "</title>", self.source)
        self.assertIn('property="og:title" content="' + TITLE + '"', self.source)
        top = self.section('top')
        for removed in ('class="subtitle"', 'class="lead"', 'class="eyebrow"',
                        'class="status-line"', 'class="note"'):
            self.assertNotIn(removed, top)
        self.assertEqual(re.findall(r'<a\b[^>]*href="([^"]+)"', top), ['#film', '#start'])
        main = self.source.split('<main id="main">', 1)[1].split('</main>', 1)[0]
        self.assertEqual(re.findall(r'<section\b[^>]*\bid="([^"]+)"', main)[:2], ['top', 'film'])

    def test_removed_sections_leave_no_dangling_navigation(self):
        for name in ("idea", "workflows", "status", "performance-notes", "slicer"):
            self.assertNotIn('id="' + name + '"', self.source)
            self.assertNotIn('href="#' + name + '"', self.source)
        self.assertNotIn("fig-route-guide.png", self.source)
        self.assertIn("workflow-overview.svg", self.source)

    def test_methods_are_the_last_section_without_separate_references(self):
        main = self.source.split('<main id="main">', 1)[1].split('</main>', 1)[0]
        sections = re.findall(r'<section\b[^>]*\bid="([^"]+)"', main)
        self.assertEqual(sections[-1], "integrated")
        self.assertNotIn('cite', sections)

    def test_header_is_navigation_only_and_footer_uses_vector_wordmark(self):
        header = self.source.split('<header class="site-header"', 1)[1].split('</header>', 1)[0]
        self.assertNotIn('<img', header)
        self.assertNotIn('class="brand"', header)
        self.assertNotIn('Private pre-release', self.source)
        self.assertNotIn('chip-status', header)
        logo = re.search(r'class="foot-brand"[^>]*>\s*<img\b([^>]+)>', self.source).group(1)
        self.assertIn('src="assets/img/slicerdspace-wordmark-light.svg"', logo)
        self.assertIn('width="1324" height="239"', logo)
        self.assertNotIn('src="assets/img/logo-ddsurfer.png"', self.source)
        record = json.loads((ROOT / "docs/website/logo_provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(record["source_repository"], "ChengjinLii/paper-latex")
        subprocess.run([sys.executable, "scripts/check_website_assets.py", "--check"],
                       cwd=ROOT, check=True, capture_output=True, text=True)

    def test_complete_results_plate_replaces_individual_gallery_tiles(self):
        section = self.section('outputs')
        self.assertNotIn('class="tile', section)
        self.assertEqual(len(re.findall(r'<img\b', section)), 1)
        self.assertRegex(section, r'src="assets/img/results-overview\.svg\?v=[^"]+"')
        self.assertIn('width="1830" height="2036"', section)
        self.assertIn('Open full-size SVG', section)
        receipt = json.loads((ROOT / "docs/website/published_assets.json").read_text())
        published = ROOT / "docs/site/assets/img/results-overview.svg"
        entry = receipt['files']['docs/site/assets/img/results-overview.svg']
        self.assertEqual(hashlib.sha256(published.read_bytes()).hexdigest(), entry['sha256'])
        self.assertEqual(entry['query_preset'], 'official_cerebellar')
        self.assertRegex(entry['query_provenance_sha256'], r'^[a-f0-9]{64}$')
        self.assertNotIn('res-wmql-csd.jpg', self.source)

    def test_integration_is_standalone_and_source_backed(self):
        integrated = self.section("integrated")
        methods = re.findall(r'<tr data-method="([^"]+)"', integrated)
        self.assertEqual(len(methods), 35)
        self.assertEqual(len(set(methods)), 35)
        for method in ("pnl_cnn", "localpca", "patch2self", "eddy", "topup", "dti",
                       "freewater", "noddi", "dki", "mkcurve", "ddseg", "ddparcel",
                       "ddsurfer", "surface_postprocess", "ukf", "saf", "wmql",
                       "wma", "tractcloud", "sift2", "scalar_length", "regional"):
            self.assertIn(method, methods)
        for row in re.findall(r'<tr data-method=.*?</tr>', integrated, re.S):
            self.assertRegex(row, r'class="method-reference"[^>]+href="https://')
        subprocess.run([sys.executable, "scripts/build_website_catalog.py", "--check"],
                       cwd=ROOT, check=True, capture_output=True, text=True)

    def test_integration_lists_resources_without_obsolete_status_badges(self):
        table = re.search(r'<table class="int-table">(.*?)</table>', self.source, re.S).group(1)
        for old in ("st-partial", "MATLAB", "no recorded run", "synthetic data only", "Partial:"):
            self.assertNotIn(old, table)
        self.assertIn("Runtime / access", table)
        self.assertIn("Python API", table)
        self.assertIn("UKF reference route", table)
        self.assertIn("full suite optional", table)

    def test_citations_are_direct_links_in_method_rows(self):
        self.assertNotIn('id="cite"', self.source)
        self.assertNotIn('ref-list', self.source)
        self.assertNotIn('href="#ref-', self.source)
        self.assertIn('Papers', self.section('integrated'))
        self.assertIn('href="https://doi.org/10.1002/advs.76596"', self.section('integrated'))
        links = re.findall(r'<a class="method-reference"[^>]*data-reference="(\d+)"[^>]*>(.*?)</a>', self.section('integrated'))
        self.assertTrue(links)
        self.assertTrue(all(text == '[' + number + ']' for number, text in links))

    def test_film_area_has_one_download_and_does_not_preload_video(self):
        film = self.section('film')
        self.assertIn('preload="none"', film)
        self.assertIn('poster="assets/img/poster.jpg"', film)
        meta = re.search(r'<p class="meta-row">(.*?)</p>', film, re.S).group(1)
        self.assertEqual(len(re.findall(r'<a\b', meta)), 2)
        self.assertIn('class="source-link"', meta)
        self.assertIn('https://github.com/NeuroDMRI/SlicerDSpace', meta)
        self.assertIn('Download video', meta)
        self.assertIn('<span>Diffusion MRI workflows in 3D Slicer.</span>', meta)
        self.assertLess(meta.index('Diffusion MRI workflows'), meta.index('Download video'))
        self.assertLess(meta.index('Diffusion MRI workflows'), meta.index('class="source-link"'))
        self.assertLess(meta.index('class="source-link"'), meta.index('Download video'))
        for removed in ('72.0 s', '1920 × 1080', '.srt', 'id="transcript"', 'narrated', 'subtitles in the picture'):
            self.assertNotIn(removed, film)

    def test_results_caption_does_not_single_out_cerebellar_queries(self):
        caption = re.search(r'<figcaption>(.*?)</figcaption>', self.section('outputs'), re.S).group(1)
        self.assertNotIn('CBL-BrS', caption)
        self.assertNotIn('CBL-CBL', caption)

    def test_references_have_primary_links_and_no_dangling_ids(self):
        catalog = json.loads((ROOT / "docs/website/references.json").read_text(encoding="utf-8"))
        used = set(re.findall(r'data-reference="(\d+)"', self.source))
        self.assertTrue(used)
        for item in catalog["references"]:
            if str(item['id']) in used:
                self.assertTrue(item["url"].startswith("https://"))
                self.assertIn(html.escape(item["url"], quote=True), self.source)
                self.assertIn(html.escape(item['citation'], quote=True), self.source)

    def test_subject_identity_does_not_dominate_main_copy(self):
        for name in ("top", "film", "outputs"):
            copy = self.section(name)
            self.assertNotIn("EDDEN sub-01", copy)
        self.assertTrue((ROOT / 'docs/website/FACTS.md').is_file())

    def test_public_facing_copy_and_footer_avoid_development_status(self):
        self.assertNotIn('pre-release', self.source.lower())
        self.assertNotIn('repository is private', self.source.lower())
        footer = self.source.split('<footer class="site-footer">', 1)[1].split('</footer>', 1)[0]
        self.assertNotIn('About the film', footer)
        self.assertNotIn('Rosetta', footer)
        self.assertIn('class="foot-nav"', footer)
        self.assertIn('LICENSE.txt', footer)
        self.assertIn('THIRD_PARTY_NOTICES.md', footer)

    def test_footer_reuses_header_gradient_and_keeps_navigation_structured(self):
        styles = (ROOT / 'docs/site/assets/css/site.css').read_text(encoding='utf-8')
        header = re.search(r'\.site-header::after\s*\{([^}]+)\}', styles).group(1)
        footer = re.search(r'\.site-footer::before\s*\{([^}]+)\}', styles).group(1)
        for shared in ('height: 2px', 'background: var(--tri)', 'opacity: .7'):
            self.assertIn(shared, header)
            self.assertIn(shared, footer)
        markup = self.source.split('<footer class="site-footer">', 1)[1].split('</footer>', 1)[0]
        self.assertIn('<a class="foot-brand" href="#top">', markup)
        self.assertIn('<nav class="foot-links" aria-label="Legal information">', markup)
        self.assertIn('<a class="foot-top-link" href="#top">', markup)
        self.assertEqual(markup.count('<ul>'), 2)

    def test_setup_describes_current_gui_and_first_execution(self):
        start = self.section("start")
        for requirement in ("--slicer-version 5.10", "Additional module paths", "UKFTractography",
                            "nodif_brain_mask.nii.gz", "Prepare selected analysis environment",
                            "Results and QC", "--doctor", "--plan", "--execute", "--jobs 2"):
            self.assertIn(requirement, start)
        self.assertNotIn("Advanced settings › Results and QC", self.source)

    def test_skills_are_within_get_started_and_link_to_existing_guides(self):
        start = self.section('start')
        section = start.split('<div class="start-skills"', 1)[1].split('<h3 class="start-route-title start-manual-title">', 1)[0]
        self.assertIn('id="skills"', section)
        self.assertNotIn('<section class="section reveal" id="skills"', self.source)
        self.assertNotIn('href="#skills">Agent Skills</a>', self.source)
        self.assertIn('href="#start">Get started</a>', self.source)
        self.assertIn('id="skills-title"', section)
        agent_steps = section.split('<ol class="start-steps agent-steps">', 1)[1].rsplit('</ol>', 1)[0]
        self.assertEqual(agent_steps.count('<li class="start-step">'), 3)
        self.assertIn('<h4>Read the Skill</h4>', agent_steps)
        self.assertIn('<h4>Install and enable</h4>', agent_steps)
        self.assertIn('<h4>Use an existing installation</h4>', agent_steps)
        self.assertNotIn('skill-capabilities', section)
        self.assertNotIn('skills-grid', section)
        self.assertIn('Show me the plan and wait for confirmation before running', section)
        self.assertIn('id="agent-install-prompt"', section)
        self.assertIn('id="agent-analysis-prompt"', section)
        self.assertNotIn('<details', section)
        for relative in ('.agents/skills/slicerdspace/SKILL.md',
                         '.agents/skills/slicerdspace/references/usage.md',
                         'docs/setup/CLI_TEMPLATES.md'):
            self.assertIn('href="https://github.com/NeuroDMRI/SlicerDSpace/blob/main/' + relative + '"', section)
            self.assertTrue((ROOT / relative).is_file())

    def test_output_heading_and_cli_are_direct_and_visible(self):
        self.assertIn('<h2>Outputs from SlicerDSpace.</h2>', self.section('outputs'))
        start = self.section('start')
        self.assertIn('<div class="start-cli" id="cli"', start)
        self.assertNotIn('<details class="transcript start-cli"', start)
        self.assertIn('--jobs 2', start)

    def test_setup_routes_have_one_heading_level_and_matching_visual_markers(self):
        start = self.section('start')
        headings = re.findall(r'<h3 class="start-route-title[^\"]*"[^>]*>(.*?)</h3>', start)
        self.assertEqual(len(headings), 3)
        for heading, marker in zip(headings, 'ABC'):
            self.assertIn(f'class="start-route-marker" aria-hidden="true">{marker}</span>', heading)
            self.assertIn('start-route-text', heading)
        self.assertIn('class="start-recommended">(Recommended)</span>', headings[0])
        styles = (ROOT / 'docs/site/assets/css/site.css').read_text(encoding='utf-8')
        self.assertNotIn('.start-skills > h3', styles)
        self.assertNotIn('.start-cli > h3', styles)
        for selector in ('.start-route-marker', '.start-recommended'):
            rule = re.search(re.escape(selector) + r'\s*\{([^}]+)\}', styles).group(1)
            self.assertIn('color: var(--route-accent)', rule)
            self.assertNotIn('var(--orange)', rule)

    def test_figure_wrappers_are_unframed_and_result_assets_remain_available(self):
        self.assertNotIn('class="panel shot', self.source)
        self.assertIn('<figure class="results-overview">', self.section('outputs'))
        for name in ('choose', 'runtime'):
            self.assertIn('<figure class="shot fig">', self.section(name))
        styles = (ROOT / 'docs/site/assets/css/site.css').read_text(encoding='utf-8')
        self.assertIn('.results-scroll { overflow-x: auto;', styles)

    def test_guide_floats_without_header_docking(self):
        self.assertIn('id="guide-bot"', self.source)
        self.assertNotIn('guide-bot__dock', self.source)
        script = (ROOT / 'docs/site/assets/js/guide-bot.js').read_text(encoding='utf-8')
        self.assertNotIn('is-docked', script)
        self.assertNotIn('positionLauncher', script)
        self.assertIn('Get started includes the Skill', script)
        for name in ('assets/css/guide-bot.css', 'assets/js/guide-bot.js', 'assets/img/guide-bot-hat.svg'):
            self.assertTrue((ROOT / 'docs/site' / name).is_file())

    def test_both_readmes_promote_installation_and_use_skills(self):
        for name, templates in (('README.md', 'CLI_TEMPLATES.md'), ('README.zh-CN.md', 'CLI_TEMPLATES.zh-CN.md')):
            with self.subTest(readme=name):
                text = (ROOT / name).read_text(encoding='utf-8')
                self.assertIn('[Agent Skills](#agent-skills)', text)
                content = text.split('## Agent Skills\n', 1)[1].split('\n---\n', 1)[0]
                self.assertIn('.agents/skills/slicerdspace/SKILL.md', content)
                self.assertIn('.agents/skills/slicerdspace/references/usage.md', content)
                self.assertIn('docs/setup/' + templates, content)
                self.assertIn('/path/to/subject', content)
                self.assertIn('/path/to/results', content)
                self.assertIn('QC', content)


if __name__ == "__main__":
    unittest.main()
