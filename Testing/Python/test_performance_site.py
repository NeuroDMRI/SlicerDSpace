"""Public measurements remain traceable and never inherit private evidence paths."""

import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from scripts.build_performance_site import build_evidence, mkcurve_signal_records, render

ROOT = Path(__file__).resolve().parents[2]


class PerformanceSiteTests(unittest.TestCase):
    def test_public_measurements_match_checked_evidence(self):
        evidence = build_evidence()
        saved = json.loads((ROOT / 'docs/site/assets/data/performance.json').read_text())
        self.assertEqual(saved, evidence)
        self.assertEqual(len(evidence['records']), 68)
        for row in evidence['records']:
            self.assertTrue(math.isclose(row['speed_ratio'], row['reference_seconds'] / row['optimized_seconds']))
            self.assertTrue(row['scope'] and row['baseline'] and row['limitation'])
            self.assertIn(row['measurement_source'], evidence['sources'])
            source = ROOT / 'docs/release/benchmarks' / row['measurement_source']['file']
            self.assertEqual(hashlib.sha256(source.read_bytes().replace(b'\r\n', b'\n')).hexdigest(), row['measurement_source']['sha256'])
        self.assertEqual(next(row for row in evidence['records'] if row['id'] == 'noddi')['category'], 'parallel')
        self.assertEqual(next(row for row in evidence['records'] if row['id'] == 'mkcurve')['category'], 'port')
        self.assertEqual(next(row for row in evidence['records'] if row['id'] == 'dki')['category'], 'integration')
        self.assertEqual(next(row for row in evidence['records'] if row['id'] == 'surface-mapping')['category'], 'inherited')
        self.assertEqual(next(row for row in evidence['records'] if row['id'] == 'dwi-io')['category'], 'io')
        self.assertTrue(all(row['upstream_reference'] for row in evidence['records']))
        self.assertEqual(evidence['upstream_review']['stable_versions']['dipy'], '1.12.1')
        serialized = json.dumps(evidence)
        for private in ('/data/', '/home/', '/Users/', '/Volumes/', 'snapshot_hashes', 'source_path'):
            self.assertNotIn(private, serialized)
        page = (ROOT / 'docs/site/index.html').read_text(encoding='utf-8')
        self.assertIn(render(evidence), page)

    def test_runtime_and_optimization_evidence_live_only_on_the_home_page(self):
        page = (ROOT / 'docs/site/index.html').read_text(encoding='utf-8')
        self.assertIn('id="runtime"', page)
        self.assertIn('id="performance"', page)
        self.assertIn('id="performance-evidence"', page)
        self.assertIn('href="#performance"', page)
        self.assertIn('assets/img/fig-runtime-resources.png', page)
        self.assertNotIn('assets/img/fig-runtime-optimizations.png', page)
        self.assertNotIn('perf-chart', page)
        self.assertNotIn('perf-filters', page)
        self.assertNotIn('id="perf-count"', page)
        self.assertIn('<div class="perf-evidence" id="performance-evidence">', page)
        self.assertNotIn('<details class="perf-evidence"', page)
        self.assertNotIn('View performance comparisons', page)
        self.assertFalse((ROOT / 'docs/site/performance.html').exists())
        for name in ('index.html', '404.html'):
            self.assertNotIn('performance.html', (ROOT / 'docs/site' / name).read_text(encoding='utf-8'))

    def test_unverified_full_brain_result_is_not_publishable(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for name in ('freewater_latest_stable_full_20261022.json', 'complete_models_20261020.json'):
                shutil.copy2(ROOT / 'docs/release/benchmarks' / name, folder / name)
            file = folder / 'freewater_latest_stable_full_20261022.json'
            record = json.loads(file.read_text())
            record['qc_equal'] = False
            file.write_text(json.dumps(record))
            with patch('scripts.build_performance_site.BENCHMARKS', folder), self.assertRaises(ValueError):
                build_evidence()

    def test_method_details_are_adjacent_rows_in_the_same_table(self):
        evidence = build_evidence()
        markup = render(evidence)
        self.assertEqual(markup.count('class="perf-record"'), 68)
        self.assertEqual(markup.count('class="perf-detail-row"'), 68)
        self.assertNotIn('perf-methods', markup)
        self.assertNotIn('Methods &amp; verification', markup)
        self.assertNotIn('<details', markup)
        for row in evidence['records']:
            before = markup.index('data-record="' + row['id'] + '"')
            main_end = markup.index('</tr>', before) + len('</tr>')
            self.assertTrue(markup[main_end:].startswith('<tr class="perf-detail-row" id="method-' + row['id'] + '" hidden>'))
            self.assertIn('<button class="perf-toggle" type="button"', markup[before:main_end])
            self.assertIn('aria-controls="method-' + row['id'] + '" aria-expanded="false"', markup[before:main_end])
            self.assertIn('<span class="perf-toggle-icon" aria-hidden="true">+</span>', markup[before:main_end])

    def test_details_start_hidden_in_both_pages_without_javascript_or_css(self):
        from html.parser import HTMLParser

        class Rows(HTMLParser):
            def __init__(self):
                super().__init__()
                self.rows = []

            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if tag == 'tr' and attrs.get('class') == 'perf-detail-row':
                    self.rows.append(attrs)

        for filename in ('index.html', 'zh.html'):
            with self.subTest(filename=filename):
                parser = Rows()
                parser.feed((ROOT / 'docs/site' / filename).read_text(encoding='utf-8'))
                self.assertEqual(len(parser.rows), 68)
                self.assertTrue(all('hidden' in row for row in parser.rows))

    def test_new_local_optimizations_require_complete_equivalence_checks(self):
        import copy
        original_read = Path.read_text
        for file, key in (('wma_transform_20261021.json', 'exact_transform'),
                          ('wma_fibers_bounded_20261021.json', 'exact_entire_optimizer_trajectory'),
                          ('freewater_dispatch_slicer_20261021.json', 'exact_raw_parameters')):
            with self.subTest(file=file):
                record = copy.deepcopy(json.loads(original_read(ROOT / 'docs/release/benchmarks' / file)))
                record['records'][0][key] = False
                def changed(path, *args, **kwargs):
                    return json.dumps(record) if path.name == file else original_read(path, *args, **kwargs)
                with patch.object(Path, 'read_text', changed), self.assertRaises(ValueError):
                    build_evidence()

    def test_latest_protocol_pair_is_selected_and_cannot_fall_back_after_failed_qc(self):
        evidence = build_evidence()
        record = next(row for row in evidence['records'] if row['id'] == 'freewater')
        self.assertEqual(record['measurement_source']['file'], 'freewater_protocol_full_20261029.json')
        self.assertEqual(record['evidence_date'], '2026-10-29')
        self.assertIn('protocol', record['method'])
        original_read = Path.read_text

        def changed(path, *args, **kwargs):
            text = original_read(path, *args, **kwargs)
            if path.name == 'freewater_protocol_full_20261029.json':
                record = json.loads(text)
                record['qc_equal'] = False
                return json.dumps(record)
            return text

        with patch.object(Path, 'read_text', changed), self.assertRaises(ValueError):
            build_evidence()

    def test_signal_reuse_is_incremental_and_keeps_both_solver_outcomes(self):
        report = json.loads((ROOT / 'docs/release/benchmarks/performance_next_20261029.json').read_text())
        records = mkcurve_signal_records(report)
        row = next(row for row in build_evidence()['records'] if row['id'] == 'mkcurve-signal')
        self.assertEqual(row['category'], 'port')
        self.assertEqual(row['evidence_date'], '2026-10-29')
        self.assertEqual(row['reference_seconds'], records[0]['reference_median_seconds'])
        self.assertEqual(len(row['solver_results']), 6)
        self.assertTrue(any(case['reduction_percent'] < 0 for case in row['solver_results']))
        self.assertIn('preceding qualified-QR', row['baseline'])
        self.assertIn('only for the reference solver', row['limitation'])
        self.assertNotIn('ddparcel', {item['id'] for item in build_evidence()['records']})

    def test_signal_reuse_rejects_incomplete_or_inconsistent_records(self):
        import copy
        report = json.loads((ROOT / 'docs/release/benchmarks/performance_next_20261029.json').read_text())
        mutations = (
            lambda r: r['mkcurve']['records'].pop(),
            lambda r: r['mkcurve']['records'][-1].update(exact_output_parity=False),
            lambda r: r['mkcurve']['records'][0].update(reference_median_seconds=.1),
            lambda r: r['mkcurve']['records'][0].update(speed_ratio=10),
            lambda r: r['mkcurve'].update(source_inputs_unchanged=False),
            lambda r: r['mkcurve']['environment']['thread_env'].update(OMP_NUM_THREADS='8'),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                candidate = copy.deepcopy(report)
                mutate(candidate)
                with self.assertRaises(ValueError):
                    mkcurve_signal_records(candidate)


if __name__ == '__main__':
    unittest.main()
