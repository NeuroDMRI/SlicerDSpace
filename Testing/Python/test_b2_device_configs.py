#!/usr/bin/env python3
"""B2 device-configuration fixes D1..D6 (host RAM warning, TLS trust, small screens, DICOM space, NHDR path, mirrors)."""

from __future__ import annotations

import contextlib
import errno
import io
import os
import re
import shutil
import ssl
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

GIB = 1024 ** 3


class D3SmallScreenTest(unittest.TestCase):
    """D3 (audit F3): the first-run wizard and the environment dialog fit a 1092x574 logical screen."""

    SMALL = (1092, 574)  # 1366x768 at 125 %
    LARGE = (1920, 1080)
    CHROME_ESTIMATE = 180  # header, progress line, footer buttons and margins around the wizard's scroll area (audit estimate)

    @staticmethod
    def _qt(screen_size):
        qt = mock.MagicMock(name="qt")
        geometry = mock.MagicMock(name="availableGeometry")
        geometry.width.return_value = screen_size[0]
        geometry.height.return_value = screen_size[1]
        geometry.left.return_value = 0
        geometry.top.return_value = 0
        screen = mock.MagicMock(name="screen")
        screen.availableGeometry.return_value = geometry
        qt.QApplication.screenAt.return_value = screen
        qt.QApplication.primaryScreen.return_value = screen
        return qt

    def _run_wizard(self, screen_size):
        from shared import ui_onboarding

        qt = self._qt(screen_size)
        slicer = mock.MagicMock(name="slicer")
        owner = mock.MagicMock(name="owner")
        owner._onboardingLanguage = "en"
        owner._read_environment_path_setting.return_value = ""
        with mock.patch.dict(sys.modules, {"qt": qt, "slicer": slicer}):
            ui_onboarding.show_first_run_wizard(owner)
        dialog = qt.QDialog.return_value
        scroll = qt.QScrollArea.return_value
        return dialog, scroll

    @staticmethod
    def _last_arg(widget, method):
        calls = getattr(widget, method).call_args_list
        return calls[-1][0] if calls else None

    def _run_environment_dialog(self, screen_size):
        from shared import ui_environment_dialog as module

        qt = self._qt(screen_size)
        slicer = mock.MagicMock(name="slicer")
        owner = mock.MagicMock(name="owner")
        owner._environmentDialog = None
        owner._onboardingLanguage = "en"
        state = mock.MagicMock(name="state")
        state.missing_items = ()
        with mock.patch.dict(sys.modules, {"qt": qt, "slicer": slicer}), \
                mock.patch.object(module, "environment_dialog_state", return_value=state):
            module.show_environment_results_dialog(
                owner=owner, result=mock.MagicMock(), python_install_target="x",
                on_install_all=lambda: None, on_show_notes=lambda: None, on_install_item=lambda item: None)
        return qt.QDialog.return_value

    def test_b2_d3_wizard_minimum_sizes_fit_a_small_screen(self):
        dialog, scroll = self._run_wizard(self.SMALL)
        (min_width,) = self._last_arg(dialog, "setMinimumWidth")
        (scroll_min,) = self._last_arg(scroll, "setMinimumHeight")
        self.assertLessEqual(min_width, self.SMALL[0])
        self.assertLessEqual(scroll_min + self.CHROME_ESTIMATE, self.SMALL[1])
        self.assertGreater(scroll_min, 0)

    def test_b2_d3_wizard_keeps_today_s_minimum_sizes_on_a_large_screen(self):
        dialog, scroll = self._run_wizard(self.LARGE)
        self.assertEqual(self._last_arg(dialog, "setMinimumWidth"), (760,))
        self.assertEqual(self._last_arg(scroll, "setMinimumHeight"), (430,))

    def test_b2_d3_environment_dialog_size_is_clamped_to_a_small_screen(self):
        dialog = self._run_environment_dialog(self.SMALL)
        width, height = self._last_arg(dialog, "resize")
        self.assertLessEqual(width, self.SMALL[0])
        self.assertLessEqual(height, self.SMALL[1])

    def test_b2_d3_environment_dialog_keeps_today_s_size_on_a_large_screen(self):
        dialog = self._run_environment_dialog(self.LARGE)
        self.assertEqual(self._last_arg(dialog, "resize"), (900, 640))

    def test_b2_d3_unknown_screen_keeps_today_s_sizes(self):
        from shared.ui_compact import fit_size_to_screen

        qt = mock.MagicMock(name="qt")
        qt.QApplication.screenAt.return_value = None
        qt.QApplication.primaryScreen.return_value = None
        self.assertEqual(fit_size_to_screen(qt, 900, 640), (900, 640))




NHDR_TEXT = "NRRD0004\ntype: short\ndimension: 3\nsizes: 2 2 2\nencoding: raw\ndata file: dwi.raw\n"


class D5DetachedNrrdPathTest(unittest.TestCase):
    """D5 (audit F5): a detached NRRD whose data path is not ASCII gives a clear message, not UnicodeEncodeError."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.header = self.root / "dwi.nhdr"
        self.header.write_text(NHDR_TEXT, encoding="ascii")

    def _config(self):
        from shared.input_import import InputImportConfig

        return InputImportConfig(source=self.header, subject_root=self.root / "subj", subject_id="s1", dry_run=True)

    def test_b2_d5_non_ascii_data_path_gives_the_clear_message_that_names_the_path(self):
        from shared.ui_prepare_inputs import import_detached_nrrd

        folder = self.root / "数据"
        folder.mkdir()
        raw = folder / "dwi.raw"
        raw.write_bytes(b"\0" * 16)
        with mock.patch("shared.input_import.prepare_inputs") as prepare:
            with self.assertRaises(ValueError) as caught:
                import_detached_nrrd(self._config(), raw)
        prepare.assert_not_called()
        message = str(caught.exception)
        self.assertIn("non-ASCII", message)
        self.assertIn("ASCII-only folder", message)
        self.assertIn(raw.as_posix(), message)

    def test_b2_d5_ascii_data_path_writes_the_same_header_bytes_as_before(self):
        from shared.ui_prepare_inputs import import_detached_nrrd

        raw = self.root / "dwi.raw"
        raw.write_bytes(b"\0" * 16)
        staged_bytes = []

        def fake_prepare(config):
            staged_bytes.append(Path(config.source).read_bytes())
            return {"plan": {}, "reused": False}

        with mock.patch("shared.input_import.prepare_inputs", side_effect=fake_prepare):
            import_detached_nrrd(self._config(), raw)
        expected = ("NRRD0004\ntype: short\ndimension: 3\nsizes: 2 2 2\nencoding: raw\n"
                    f"data file: {raw.as_posix()}\n").encode("ascii")  # pinned from the base code's output
        self.assertEqual(staged_bytes, [expected])

    def test_b2_d5_the_preview_staging_keeps_its_message(self):
        from shared.input_preview import PreviewError, stage_detached_header

        folder = self.root / "数据"
        folder.mkdir()
        raw = folder / "dwi.raw"
        raw.write_bytes(b"\0" * 16)
        with self.assertRaises(PreviewError) as caught:
            stage_detached_header(self.header, raw, self.root)
        self.assertIn("non-ASCII characters; move the file to an ASCII-only folder", caught.exception.detail)


class D1ParcellationRamWarningTest(unittest.TestCase):
    """D1: DDParcel checks real available host RAM against the 7 GiB floor (no 0.7x scheduling budget or 8 GiB fallback)."""

    def _run_ddparcel_with_ram(self, available_ram):
        from shared.orchestrator import ProductOrchestratorConfig, run_subject_products
        import test_product_orchestrator as T

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "sub1"
            T._write_valid_raw_subject(root, shape=(2, 2, 2), affine=T.UNAFFECTED_AFFINE)
            atlas_t2 = Path(td) / "atlas_t2.nii.gz"
            atlas_t2.write_bytes(b"fake")
            logs = []
            cfg = ProductOrchestratorConfig(
                subject_id="sub1",
                subject_root=root,
                targets=("parcellation",),
                execute=True,
                resume=False,
                atlas_t2=atlas_t2,
            )
            reg_manifest = Path(td) / "reg_manifest.json"
            reg_manifest.write_text("{}", encoding="utf-8")
            fake_registered = mock.MagicMock(manifest_path=reg_manifest)
            fake_parcel = mock.MagicMock()
            fake_parcel.outputs = {}
            fake_parcel.manifest_path = None
            fake_parcel.qc_path = None
            with mock.patch("shared.platform_info.available_host_ram_bytes", return_value=available_ram), \
                    mock.patch("shared.orchestrator._parcellation_complete", return_value=False), \
                    mock.patch("shared.pipeline.ensure_registered_dti_bundle", return_value=fake_registered), \
                    mock.patch("anatomy.parcellation.run_ddparcel_from_registered_bundle", return_value=fake_parcel), \
                    mock.patch("shared.orchestrator._publish_parcellation_outputs", return_value={}):
                run_subject_products(cfg, log=logs.append)
            return logs

    def test_b2_d1_warns_with_real_available_ram_below_7_gib(self):
        logs = self._run_ddparcel_with_ram(int(5.2 * GIB))
        warnings = [line for line in logs if "WARNING: Available host memory for DDParcel is" in line]
        self.assertGreaterEqual(len(warnings), 1, logs)
        self.assertIn("5.2 GiB", warnings[0])
        self.assertIn("5.5 GiB", warnings[0])

    def test_b2_d1_does_not_false_warn_between_7_and_10_gib_or_when_unknown(self):
        for available in (8 * GIB, None):
            with self.subTest(available=available):
                logs = self._run_ddparcel_with_ram(available)
                self.assertFalse(
                    [line for line in logs if "Available host memory for DDParcel" in line or "WARNING: Available RAM" in line],
                    logs,
                )


class D2TlsTrustStoreTest(unittest.TestCase):
    """D2: strict SAN-only TLS context trusting system roots + SLICERDSPACE_CA_BUNDLE + certifi, including HTTPS proxies."""

    def test_b2_d2_ssl_context_disables_common_name_fallback_and_accepts_ca_bundle_env(self):
        from shared import network_config as nc

        with tempfile.TemporaryDirectory() as td:
            bundle = Path(td) / "corp-ca.pem"
            bundle.write_text("placeholder", encoding="utf-8")
            with mock.patch.dict(os.environ, {nc.CA_BUNDLE_ENV: str(bundle)}, clear=False):
                self.assertEqual(nc._explicit_cafile(), str(bundle))
                fake_ctx = ssl.create_default_context()
                with mock.patch.object(nc.ssl, "create_default_context", return_value=fake_ctx), \
                        mock.patch.object(fake_ctx, "load_verify_locations") as load_ca:
                    ctx = nc._ssl_context()
                self.assertIs(ctx, fake_ctx)
                self.assertFalse(ctx.hostname_checks_common_name)
                loaded_files = [call.kwargs.get("cafile") for call in load_ca.call_args_list]
                self.assertIn(str(bundle), loaded_files)

    def test_b2_d2_http_session_passes_ssl_context_and_proxy_ssl_context_for_https_proxies(self):
        from shared import network_config as nc

        session = nc._make_http_session("https://127.0.0.1:8443")
        self.assertIsNotNone(session)
        self.assertIs(session.verify, True)
        adapter = session.get_adapter("https://example.com")
        pool_ctx = adapter.poolmanager.connection_pool_kw.get("ssl_context")
        self.assertIsInstance(pool_ctx, ssl.SSLContext)
        self.assertFalse(pool_ctx.hostname_checks_common_name)
        proxy_mgr = adapter.proxy_manager_for("https://127.0.0.1:8443")
        self.assertIs(proxy_mgr.connection_pool_kw.get("ssl_context"), pool_ctx)
        self.assertIs(proxy_mgr.proxy_ssl_context, pool_ctx)


class D4DicomCrossVolumeSpaceTest(unittest.TestCase):
    """D4: cross-volume DICOM staging checks free space only for files that actually failed os.link."""

    def test_b2_d4_partial_link_checks_free_space_only_for_unlinked_files(self):
        from shared.input_import import InputImportConfig
        from shared.ui_prepare_inputs import import_dicom_series

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "dcm"
            src.mkdir()
            f1 = src / "s1.dcm"
            f2 = src / "s2.dcm"
            f1.write_bytes(b"a" * 8000)
            f2.write_bytes(b"b" * 2000)
            real_link = os.link

            def partial_link(a, b):
                if Path(a).name == "s1.dcm":
                    return real_link(a, b)
                raise OSError(errno.EXDEV, "cross-device link")

            cfg = InputImportConfig(source=src, subject_root=root / "subj", subject_id="sub1", dry_run=True)
            # Free space is 3000 bytes: smaller than full series (10000), larger than the one unlinked file (2000).
            usage = shutil._ntuple_diskusage(100000, 97000, 3000)
            with mock.patch("os.link", side_effect=partial_link), \
                    mock.patch("shutil.disk_usage", return_value=usage), \
                    mock.patch("shared.input_import.prepare_inputs", return_value={"plan": {}, "reused": False}) as prep:
                import_dicom_series(cfg, {"uid": "1.2.3", "paths": [f1, f2]})
            prep.assert_called_once()

    def test_b2_d4_insufficient_space_for_unlinked_files_raises_enospc_before_copy(self):
        from shared.input_import import InputImportConfig
        from shared.ui_prepare_inputs import import_dicom_series

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "dcm"
            src.mkdir()
            f1 = src / "s1.dcm"
            f1.write_bytes(b"a" * 4096)
            cfg = InputImportConfig(source=src, subject_root=root / "subj", subject_id="sub1", dry_run=True)
            usage = shutil._ntuple_diskusage(10000, 9000, 1000)
            with mock.patch("os.link", side_effect=OSError(errno.EXDEV, "cross-device")), \
                    mock.patch("shutil.disk_usage", return_value=usage), \
                    mock.patch("shutil.copyfile") as copyfile:
                with self.assertRaises(OSError) as caught:
                    import_dicom_series(cfg, {"uid": "1.2.3", "paths": [f1]})
            self.assertEqual(caught.exception.errno, errno.ENOSPC)
            copyfile.assert_not_called()


class D6ManagedRuntimeMirrorFallbackTest(unittest.TestCase):
    """D6: managed-runtime mirror downloads verify per-candidate SHA-256/size, fall back on corrupt 200, and cap pre-write."""

    def test_b2_d6_corrupt_http_200_mirror_falls_back_to_verified_direct_url(self):
        import hashlib
        from shared import managed_runtimes as mr

        good_bytes = b"VALID-RUNTIME-ARCHIVE" * 64
        bad_bytes = b"HTML-ERROR-PAGE-200-OK"
        good_sha = hashlib.sha256(good_bytes).hexdigest()
        seen_urls = []

        class _Resp(io.BytesIO):
            def __init__(self, data):
                super().__init__(data)
                self.headers = {"Content-Length": str(len(data))}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

        def fake_open(url, timeout=60):
            seen_urls.append(url)
            return _Resp(bad_bytes if "mirror.example" in url else good_bytes)

        with tempfile.TemporaryDirectory() as td:
            part = Path(td) / "archive.part"
            candidates = ["https://mirror.example/https://github.com/a/b.tar.gz", "https://github.com/a/b.tar.gz"]
            with mock.patch("shared.network_config.expand_candidate_urls", return_value=candidates), \
                    mock.patch("shared.network_config.open_url", side_effect=fake_open):
                mr.download_verified_with_mirror_fallback(
                    "https://github.com/a/b.tar.gz",
                    part,
                    expected_sha256=good_sha,
                    expected_size=len(good_bytes),
                )
            self.assertEqual(seen_urls, candidates)
            self.assertEqual(part.read_bytes(), good_bytes)

    def test_b2_d6_max_bytes_aborts_before_writing_past_cap_for_content_length_and_streaming(self):
        from shared import managed_runtimes as mr

        class _ChunkResp:
            def __init__(self, chunks, headers=None):
                self._chunks = list(chunks)
                self.headers = headers or {}

            def read(self, _n=-1):
                return self._chunks.pop(0) if self._chunks else b""

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        with tempfile.TemporaryDirectory() as td:
            part = Path(td) / "capped.part"
            # Case 1: Content-Length header exceeds cap -> fails before opening/writing file
            with mock.patch("shared.network_config.expand_candidate_urls", return_value=["https://example.com/a.whl"]), \
                    mock.patch("shared.network_config.open_url",
                               return_value=_ChunkResp([b"x" * 200], headers={"Content-Length": "200"})):
                with self.assertRaisesRegex(ValueError, "bounded size"):
                    mr.download_verified_with_mirror_fallback(
                        "https://example.com/a.whl", part, expected_sha256="0" * 64, max_bytes=100,
                    )
            self.assertFalse(part.exists())

            # Case 2: No Content-Length -> streaming check stops before writing chunk past cap
            writes = []
            real_open = Path.open

            class _TrackFile:
                def __init__(self, fh):
                    self._fh = fh

                def write(self, data):
                    writes.append(len(data))
                    return self._fh.write(data)

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return self._fh.__exit__(*args)

            with mock.patch("shared.network_config.expand_candidate_urls", return_value=["https://example.com/a.whl"]), \
                    mock.patch("shared.network_config.open_url",
                               return_value=_ChunkResp([b"a" * 60, b"b" * 60])), \
                    mock.patch.object(Path, "open", lambda p, mode="r", *a, **k: _TrackFile(real_open(p, mode, *a, **k)) if p == part and "w" in mode else real_open(p, mode, *a, **k)):
                with self.assertRaisesRegex(ValueError, "bounded size"):
                    mr.download_verified_with_mirror_fallback(
                        "https://example.com/a.whl", part, expected_sha256="0" * 64, max_bytes=100,
                    )
            self.assertEqual(sum(writes), 60)
            self.assertFalse(part.exists())


if __name__ == "__main__":
    unittest.main()
