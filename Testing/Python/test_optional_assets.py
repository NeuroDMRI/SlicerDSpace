"""Optional resource integrity, local imports and restricted export policies."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import zipfile

from shared import optional_assets
from shared.model_asset_integrity import ModelAssetIntegrityError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from download_model_assets import load_manifest, resolve_asset_plan, _url_candidates_for_asset
from export_model_asset_mirror import export_asset_mirror
from import_external_resources import import_file, extract_org
from scripts import setup_optional_assets
from shared.selected_setup import build_setup_plan
from types import SimpleNamespace


class OptionalAssetsTests(unittest.TestCase):
    def test_real_http_download_of_selected_resource_and_network_free_reuse(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import threading
        import urllib.request
        hits = []
        data = b"small test resource, not an algorithm model"
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                hits.append(self.path)
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            def log_message(self, *_):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                relative = "brain_mask/synthstrip/synthstrip.1.pt"
                catalog = root / "catalog.json"
                catalog.write_text(json.dumps({"profiles": {"synthstrip": {"directory": "brain_mask/synthstrip", "required_paths": [relative]}},
                    "asset_records": {relative: {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data),
                        "urls": [f"http://127.0.0.1:{server.server_port}/model"]}}}))
                local_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with patch.object(setup_optional_assets, "MANIFEST", catalog), patch.object(optional_assets, "MANIFEST", catalog), \
                        patch("scripts.download_model_assets.urllib.request.urlopen", side_effect=local_opener.open):
                    for _ in range(2):
                        result = setup_optional_assets.prepare(["synthstrip"], asset_root=root / "assets", execute=True, accept_terms=True)
                        self.assertEqual(result["status"], "completed")
                    self.assertEqual((root / "assets" / relative).read_bytes(), data)
                    self.assertEqual(hits, ["/model"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_only_selected_missing_resources_and_backend_sources_are_planned(self):
        def item(key, group):
            return SimpleNamespace(key=key, kind="path", status="missing", required=True, targets=("wma",),
                                   label=key, details={}, install_group=group, install_hint="configure")
        result = SimpleNamespace(statuses=[item("wma_org_atlas", "wma"), item("wma_root", "wma"),
                                           item("tractcloud_data_dir", "tractcloud")])
        plan = build_setup_plan(result, brain_mask_backend="provided", system="Linux", machine="x86_64")
        self.assertEqual(plan["optional_asset_profiles"], ["wma", "tractcloud"])
        self.assertEqual(plan['backend_profiles'], ['wma'])
        self.assertEqual(plan['manual_items'], [])
        self.assertGreater(plan["optional_asset_size_bytes"], 1800000000)
        self.assertGreaterEqual(plan['optional_asset_peak_disk_bytes'], plan['optional_asset_size_bytes'] + 2439318525)
        clean = build_setup_plan(SimpleNamespace(statuses=[]))
        self.assertEqual(clean["optional_asset_profiles"], [])

    def _weights_gap_report(self):
        gap = SimpleNamespace(key="brain_mask_backend", kind="setting", status="missing", required=True, targets=("microstructure",),
                              label="mask", details={"backend": "pnl_cnn", "missing_required": ["weights.json"]},
                              install_group="brain_mask", install_hint="")
        return SimpleNamespace(statuses=[gap], targets=("microstructure",), blocker_count=1)

    def test_f4_missing_optional_profiles_resolves_auto_for_the_platform_without_a_route(self):
        """F4: no pnl_route means auto: native on macOS and Windows, managed on Linux (no silent managed default)."""
        report = self._weights_gap_report()
        for system, expected in (("Darwin", ["pnl-cnn"]), ("Windows", ["pnl-cnn"]), ("Linux", [])):
            with self.subTest(system=system), patch("platform.system", return_value=system):
                self.assertEqual(optional_assets.missing_optional_profiles(report, "pnl_cnn"), expected)
        with patch("platform.system", return_value="Linux"):
            self.assertEqual(optional_assets.missing_optional_profiles(report, "pnl_cnn", pnl_route="native"), ["pnl-cnn"])

    def test_f4_offline_assets_helper_passes_the_route_of_the_parameter_node(self):
        """F4: use_offline_assets resolves the node's pnlMaskRoute, so a native user gets the weights offline."""
        import shared.ui_selected_setup as ui
        qt = MagicMock()
        qt.QFileDialog.getExistingDirectory.return_value = "/offline"
        slicer = SimpleNamespace(util=SimpleNamespace(mainWindow=lambda: None, infoDisplay=lambda *args: None))
        for system, route, expected in (("Linux", "native", ["pnl-cnn"]), ("Darwin", "auto", ["pnl-cnn"]),
                                        ("Linux", "auto", [])):
            with self.subTest(system=system, route=route), tempfile.TemporaryDirectory() as tmp, \
                    patch.dict("sys.modules", {"qt": qt, "slicer": slicer}), patch("platform.system", return_value=system), \
                    patch("shared.user_config.default_user_config_path", return_value=Path(tmp) / "config.json"), \
                    patch("shared.ui_setup_jobs.start_setup_job", return_value=True) as start:
                node = SimpleNamespace(brainMaskBackend="pnl_cnn", pnlMaskRoute=route, wmaAtlas="org")
                owner = SimpleNamespace(_parameterNode=node, _lastEnvironmentResult=self._weights_gap_report(),
                                        _slicer_launcher_program=lambda: "Slicer")
                ui.use_offline_assets(owner)
                if not expected:
                    start.assert_not_called()
                    continue
                args = start.call_args[0][2][0][1]
                self.assertEqual(args[args.index("--optional-profiles") + 1:], expected)

    def test_explicit_wma_and_legacy_environment_precede_shared_directory(self):
        with patch.object(optional_assets, "verified_resource_directory") as check, \
                patch.dict("os.environ", {"WMA_ATLAS_DIR": "/old/atlas"}, clear=True):
            self.assertEqual(optional_assets.resolve_wma_atlas_directory(Path("/explicit")), Path("/explicit"))
            self.assertEqual(optional_assets.resolve_wma_atlas_directory(), Path("/old/atlas"))
            check.assert_not_called()

    def test_planning_and_missing_terms_do_not_download_or_create_directories(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(setup_optional_assets, "download_asset_plan") as download:
            root = Path(tmp) / "absent"
            plan = setup_optional_assets.prepare(["tractcloud"], asset_root=root)
            self.assertEqual(plan["status"], "planned")
            self.assertEqual(plan["profiles"], ["tractcloud"])
            with self.assertRaises(PermissionError):
                setup_optional_assets.prepare(["tractcloud"], asset_root=root, execute=True)
            download.assert_not_called()
            self.assertFalse(root.exists())

    def test_private_mirror_fails_before_partial_public_download_or_path_changes(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(setup_optional_assets, "download_asset_plan") as download:
            root = Path(tmp) / "absent"
            catalog = load_manifest(setup_optional_assets.MANIFEST)
            catalog["asset_sources"]["optional_models_mirror"]["access"] = "authorized-private"
            with patch.object(setup_optional_assets, "load_manifest", return_value=catalog), \
                    self.assertRaisesRegex(PermissionError, "mirror is private"):
                setup_optional_assets.prepare(["wma"], asset_root=root, execute=True, accept_terms=True)
            self.assertFalse(root.exists())
            download.assert_not_called()

    def test_offline_import_only_selected_files_and_refuses_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = root / "catalog.json"
            relative = "brain_mask/synthstrip/synthstrip.1.pt"
            catalog.write_text(json.dumps({"profiles": {"synthstrip": {"directory": "brain_mask/synthstrip", "required_paths": [relative]}},
                "asset_records": {relative: {"sha256": hashlib.sha256(b"author").hexdigest(), "size_bytes": 6}}}))
            source, target = root / "source", root / "target"
            (source / relative).parent.mkdir(parents=True)
            (source / relative).write_bytes(b"author")
            (source / "unselected_training_data").write_bytes(b"never copy")
            with patch.object(setup_optional_assets, "MANIFEST", catalog), patch.object(optional_assets, "MANIFEST", catalog), \
                    patch.object(setup_optional_assets, "download_asset_plan") as download:
                result = setup_optional_assets.prepare(["synthstrip"], asset_root=target, execute=True, offline_root=source)
                self.assertEqual(result["status"], "completed")
                self.assertFalse((target / "unselected_training_data").exists())
                self.assertEqual((source / relative).read_bytes(), b"author")
                (target / relative).write_bytes(b"broken")
                with self.assertRaisesRegex(ValueError, "Existing resource differs"):
                    setup_optional_assets.prepare(["synthstrip"], asset_root=target, execute=True, accept_terms=True)
                self.assertEqual((target / relative).read_bytes(), b"broken")
                download.assert_not_called()

    def test_wma_archives_and_directories_alone_are_not_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            index = root / "bundles/wma/extracted_resource_index.json"
            index.parent.mkdir(parents=True)
            member = b"actual atlas"
            index.write_text(json.dumps({"files": [{"path": "ORG-800FC-100HCP/cluster.vtp",
                "sha256": hashlib.sha256(member).hexdigest(), "size_bytes": len(member)}]}))
            catalog = root / "catalog.json"
            relative = index.relative_to(root).as_posix()
            catalog.write_text(json.dumps({"profiles": {"wma": {"directory": "bundles/wma", "required_paths": [relative]}},
                "asset_records": {relative: {"sha256": hashlib.sha256(index.read_bytes()).hexdigest(), "size_bytes": index.stat().st_size}}}))
            with patch.object(optional_assets, "MANIFEST", catalog):
                self.assertIsNone(optional_assets.verified_resource_directory("wma", asset_root=root))
                path = root / "bundles/wma/ORG-800FC-100HCP/cluster.vtp"
                path.parent.mkdir()
                path.write_bytes(member)
                self.assertEqual(optional_assets.verified_resource_directory("wma", asset_root=root), index.parent)
                path.write_bytes(b"changed")
                with self.assertRaises(ModelAssetIntegrityError):
                    optional_assets.verified_resource_directory("wma", asset_root=root)

    def test_profiles_keep_restricted_template_out_of_mirror(self):
        manifest = load_manifest(ROOT / 'models/external_assets_manifest.json')
        mirror = resolve_asset_plan(manifest, 'mirrorable-inference')
        pnl = resolve_asset_plan(manifest, 'pnl-cnn')
        template = 'brain_mask/pnl_cnn_torch/IITmean_b0_256.nii.gz'
        self.assertNotIn(template, mirror.required_paths)
        self.assertIn(template, pnl.required_paths)
        self.assertTrue(any(path.endswith('ORG-RegAtlas-100HCP.zip') for path in mirror.required_paths))
        self.assertTrue(all(mirror.asset_records[path]['redistribution'] == 'with-notice' for path in mirror.required_paths))
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'mirror'
            with self.assertRaisesRegex(RuntimeError, 'redistribution'):
                export_asset_mirror(repo_root=Path(tmp), output_dir=target, profile='pnl-cnn',
                    manifest_path=ROOT / 'models/external_assets_manifest.json', mode='copy')
            self.assertFalse(target.exists())

    def test_upstream_only_ignores_mirror_and_base_url_overrides(self):
        manifest = load_manifest(ROOT / 'models/external_assets_manifest.json')
        plan = resolve_asset_plan(manifest, 'pnl-cnn')
        record = plan.asset_records['brain_mask/pnl_cnn_torch/IITmean_b0_256.nii.gz']
        candidates = _url_candidates_for_asset(manifest, dict(record, mirrors=['https://unauthorized.invalid/template']),
            source_ids=('optional_models_mirror',), base_urls=('https://unauthorized.invalid',), allow_planned_sources=True)
        self.assertEqual(candidates, [('url', record['urls'][0])])

    def test_current_pnl_profiles_select_only_converted_weights(self):
        manifest = load_manifest(ROOT / 'models/external_assets_manifest.json')
        for profile in ('pnl-cnn', 'pnl-cnn-weights', 'mirrorable-inference'):
            plan = resolve_asset_plan(manifest, profile)
            pnl = [p for p in plan.required_paths if p.startswith('brain_mask/pnl_')]
            self.assertTrue(any(p.endswith('axial.pt') for p in pnl))
            self.assertFalse(any(p.endswith('.h5') or p.endswith('CompNetBasicModel.json') for p in pnl))
            self.assertTrue(all(p.startswith('brain_mask/pnl_cnn_torch/') for p in pnl))
        legacy = resolve_asset_plan(manifest, 'pnl-cnn-legacy')
        self.assertEqual(sum(p.endswith('.h5') for p in legacy.required_paths), 3)

    def test_pnl_preparation_download_plan_never_requests_legacy_hdf5(self):
        from scripts.download_model_assets import AssetPlan
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'assets'
            with patch.object(setup_optional_assets, 'download_asset_plan') as download, \
                    patch.object(setup_optional_assets, 'verified_resource_directory', side_effect=[None, root / 'brain_mask/pnl_cnn_torch']), \
                    patch.object(setup_optional_assets, 'verify_asset_plan', return_value={'ok': True}):
                result = setup_optional_assets.prepare(['pnl-cnn'], asset_root=root, execute=True, accept_terms=True)
            self.assertEqual(result['external_dependencies']['pnl_mask_model_dir'], str(root / 'brain_mask/pnl_cnn_torch'))
            plan = download.call_args.args[1]
            self.assertIsInstance(plan, AssetPlan)
            self.assertFalse(any(p.endswith('.h5') for p in plan.required_paths))
            self.assertEqual(sum(p.endswith('.pt') for p in plan.required_paths), 3)

    def test_pnl_prefix_import_selects_torch_without_reading_legacy_hdf5(self):
        from scripts import import_external_resources as importer
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prefix, target = root / 'prefix', root / 'assets'
            models = prefix / 'pnl-models'
            models.mkdir(parents=True)
            (models / 'weights.json').write_text('{}')
            (models / 'axial.pt').write_bytes(b'converted')
            license = prefix / 'lib/python3.11/site-packages/pnl_brain_mask-0.1.0.dist-info/licenses/LICENSE'
            license.parent.mkdir(parents=True)
            license.write_bytes(b'preserved PNL agreement')
            records = {}
            for relative, data in (
                ('brain_mask/pnl_cnn_torch/axial.pt', b'converted'),
                ('brain_mask/pnl_cnn_torch/LICENSE.txt', b'preserved PNL agreement'),
                ('brain_mask/pnl_cnn/legacy.h5', b'never acquired'),
            ):
                records[relative] = dict(size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), redistribution='with-notice')
            catalog = root / 'catalog.json'
            catalog.write_text(json.dumps({'asset_records': records}))
            with patch.object(importer, 'MANIFEST', catalog), patch.object(sys, 'argv',
                    ['import', '--asset-root', str(target), '--pnl-prefix', str(prefix), '--execute']):
                importer.main()
            self.assertEqual((target / 'brain_mask/pnl_cnn_torch/axial.pt').read_bytes(), b'converted')
            self.assertFalse((target / 'brain_mask/pnl_cnn/legacy.h5').exists())

    def test_in_place_optional_export_does_not_delete_assets_or_overwrite_core_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / 'weights/model.pt'
            model.parent.mkdir()
            model.write_bytes(b'author')
            core = root / 'asset_mirror_manifest.json'
            core.write_text('existing core metadata')
            manifest = root / 'catalog.json'
            manifest.write_text(json.dumps({'profiles': {'test': {'required_paths': ['weights/model.pt']}},
                'asset_records': {'weights/model.pt': {'sha256': hashlib.sha256(b'author').hexdigest(), 'size_bytes': 6}}}))
            export_asset_mirror(repo_root=root, output_dir=root, profile='test', manifest_path=manifest,
                                mode='copy', index_prefix='optional_assets_mirror')
            self.assertEqual(model.read_bytes(), b'author')
            self.assertEqual(core.read_text(), 'existing core metadata')
            self.assertTrue((root / 'optional_assets_mirror_manifest.json').is_file())
            with self.assertRaises(ValueError):
                export_asset_mirror(repo_root=root, output_dir=root, profile='test', manifest_path=manifest,
                                    mode='copy', index_prefix='../unsafe')

    def test_import_retains_source_refuses_tampering_and_reuses_identical_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source, target = root / 'source', root / 'cache/model'
            source.write_bytes(b'author resource')
            expected = {'sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'size_bytes': source.stat().st_size}
            self.assertEqual(import_file(source, target, expected), 'imported')
            self.assertTrue(source.is_file())
            self.assertEqual(import_file(source, target, expected), 'reused')
            target.write_bytes(b'changed')
            with self.assertRaises(FileExistsError):
                import_file(source, target, expected)
            source.write_bytes(b'wrong')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                import_file(source, target, expected)

    def test_cli_asset_root_below_a_symlinked_directory_is_accepted(self):
        # macOS keeps /tmp, /var and the default temp folder behind symlinks.
        import contextlib
        import io
        import import_external_resources as importer
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            (base / 'real').mkdir()
            (base / 'link').symlink_to(base / 'real')
            source = base / 'synthstrip.pt'
            source.write_bytes(b'author resource')
            relative = 'brain_mask/synthstrip/synthstrip.1.pt'
            catalog = base / 'catalog.json'
            catalog.write_text(json.dumps({'asset_records': {relative: {
                'sha256': hashlib.sha256(b'author resource').hexdigest(), 'size_bytes': 15, 'redistribution': 'test'}}}))
            argv = ['import_external_resources.py', '--asset-root', str(base / 'link/assets'),
                    '--synthstrip-model', str(source), '--execute']
            with patch.object(importer, 'MANIFEST', catalog), patch.object(sys, 'argv', argv), \
                    contextlib.redirect_stdout(io.StringIO()):
                importer.main()
            self.assertEqual((base / 'real/assets' / relative).read_bytes(), b'author resource')

    def test_org_extraction_keeps_structure_omits_macos_and_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            archive = root / 'ORG-800FC-100HCP.zip'
            with zipfile.ZipFile(archive, 'w') as zipped:
                zipped.writestr('ORG-800FC-100HCP/atlas.p', b'reference, never deserialized by importer')
                zipped.writestr('__MACOSX/ORG-800FC-100HCP/._atlas.p', b'not runtime data')
            records = extract_org(archive, root / 'assets')
            self.assertEqual(len(records), 1)
            self.assertFalse((root / 'assets/__MACOSX').exists())
            self.assertEqual(len(extract_org(archive, root / 'assets')), 1)
            with zipfile.ZipFile(archive, 'w') as zipped:
                zipped.writestr('../outside', b'bad')
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                extract_org(archive, root / 'unsafe')
            self.assertFalse((root / 'outside').exists())

    def test_n1_guard_extract_org_refuses_a_symlinked_destination_and_accepts_a_symlink_at_or_above_its_parent(self):
        # Guard test: it keeps the N1 rule as it is; it does not cover a new behaviour.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            archive = root / 'ORG-800FC-100HCP.zip'
            with zipfile.ZipFile(archive, 'w') as zipped:
                zipped.writestr('ORG-800FC-100HCP/atlas.p', b'reference')
            (root / 'real').mkdir()
            (root / 'link').symlink_to(root / 'real', target_is_directory=True)
            # A symlink at or above the destination's parent folder is allowed.
            self.assertEqual(len(extract_org(archive, root / 'link/assets')), 1)
            self.assertEqual((root / 'real/assets/ORG-800FC-100HCP/atlas.p').read_bytes(), b'reference')
            # The destination folder itself as a symlink is refused, and nothing is written outside.
            (root / 'outside').mkdir()
            (root / 'dest').symlink_to(root / 'outside', target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'Refusing symlink'):
                extract_org(archive, root / 'dest')
            self.assertEqual(list((root / 'outside').iterdir()), [])

    def test_shared_directory_requires_all_recorded_files_and_valid_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            path = root / 'weights/model.h5'
            path.parent.mkdir()
            manifest = root / 'catalog.json'
            manifest.write_text(json.dumps({'profiles': {'test': {'directory': 'weights', 'required_paths': ['weights/model.h5']}},
                'asset_records': {'weights/model.h5': {'sha256': hashlib.sha256(b'author').hexdigest()}}}))
            with patch.object(optional_assets, 'MANIFEST', manifest), patch.object(optional_assets, '_persist_sha256_cache'):
                self.assertIsNone(optional_assets.verified_resource_directory('test', asset_root=root))
                path.write_bytes(b'author')
                self.assertEqual(optional_assets.verified_resource_directory('test', asset_root=root), root / 'weights')
                path.write_bytes(b'tampered')
                with self.assertRaises(ModelAssetIntegrityError):
                    optional_assets.verified_resource_directory('test', asset_root=root)


if __name__ == '__main__':
    unittest.main()
