"""Unit tests for the incremental code-overlay updater and Log-below UI bar."""

from __future__ import annotations

import ast
import hashlib
import http.server
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_repackage_module():
    path = REPO_ROOT / ".github" / "scripts" / "repackage_release_assets.py"
    spec = importlib.util.spec_from_file_location("repackage_release_assets_updater_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _copy_minimal_install(dest: Path, *, extra_files: dict[str, bytes] | None = None) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    # Create a self-contained mini runtime_resources tree inside dest.
    spec = {
        "version": 1,
        "files": [
            "SlicerDSpace.py",
            "scripts/extension_runtime/runtime_resources.py",
            "scripts/extension_runtime/runtime_resources.json",
        ],
        "trees": [
            {"root": "shared", "patterns": ["*.py"]},
        ],
        "excluded_components": ["__pycache__", ".git"],
        "excluded_prefixes": [],
        "excluded_globs": [],
    }
    defaults: dict[str, bytes] = {
        "SlicerDSpace.py": b"# entry v1\nVALUE = 1\n",
        "shared/__init__.py": b"",
        "shared/helper.py": b"def version():\n    return 'v1'\n",
        "shared/obsolete.py": b"# will be removed in v2\n",
        "scripts/extension_runtime/runtime_resources.py": (
            REPO_ROOT / "scripts/extension_runtime/runtime_resources.py"
        ).read_bytes(),
        "scripts/extension_runtime/runtime_resources.json": json.dumps(spec, indent=2).encode("utf-8"),
    }
    if extra_files:
        defaults.update(extra_files)
    for rel, content in defaults.items():
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return dest


def _build_tar_gz(archive_path: Path, files: dict[str, bytes], *, symlink_member: str | None = None, traversal_member: str | None = None) -> tuple[int, str]:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz", format=tarfile.PAX_FORMAT) as tf:
        for rel, data in sorted(files.items()):
            info = tarfile.TarInfo(name=rel)
            info.size = len(data)
            info.mode = 0o755 if rel.endswith(".sh") else 0o644
            tf.addfile(info, io.BytesIO(data))
        if symlink_member is not None:
            sym = tarfile.TarInfo(name=symlink_member)
            sym.type = tarfile.SYMTYPE
            sym.linkname = "/etc/passwd"
            tf.addfile(sym)
        if traversal_member is not None:
            bad = tarfile.TarInfo(name=traversal_member)
            data = b"pwned"
            bad.size = len(data)
            tf.addfile(bad, io.BytesIO(data))
    size = archive_path.stat().st_size
    sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    return size, sha


class CodeUpdaterTests(unittest.TestCase):
    def test_tree_hash_deterministic_and_sensitive_to_edits(self) -> None:
        from shared.code_updater import compute_code_tree_sha256

        with tempfile.TemporaryDirectory() as td:
            root = _copy_minimal_install(Path(td) / "inst")
            h1, files1 = compute_code_tree_sha256(root)
            h2, files2 = compute_code_tree_sha256(root)
            self.assertEqual(h1, h2)
            self.assertEqual(files1, files2)

            (root / "shared/helper.py").write_bytes(b"def version():\n    return 'v2'\n")
            h3, _ = compute_code_tree_sha256(root)
            self.assertNotEqual(h1, h3)

    def test_repackage_build_code_overlay_matches_repo_tree_sha256(self) -> None:
        from shared.code_updater import RUNTIME_ABI_VERSION, compute_code_tree_sha256

        mod = _load_repackage_module()
        repo_sha, repo_files = compute_code_tree_sha256(REPO_ROOT)
        with tempfile.TemporaryDirectory() as td:
            manifest, paths = mod.build_code_overlay(Path(td), release_tag="v0.1.1")
            overlay_path, manifest_path = paths
            self.assertTrue(overlay_path.is_file())
            self.assertTrue(manifest_path.is_file())
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["runtime_abi_version"], RUNTIME_ABI_VERSION)
            self.assertEqual(manifest["code_sha256"], repo_sha)
            self.assertEqual(manifest["file_count"], len(repo_files))
            self.assertEqual(manifest["overlay"]["filename"], "SlicerDSpace-v0.1.1-CodeOverlay.tar.gz")
            self.assertLess(manifest["overlay"]["size_bytes"], 10 * 1024 * 1024)
            # No docs/website files leak into the overlay tree.
            self.assertFalse(any(p.startswith("docs/website/") for p in repo_files))

    def test_check_for_code_update_statuses(self) -> None:
        from shared.code_updater import RUNTIME_ABI_VERSION, check_for_code_update, compute_code_tree_sha256

        with tempfile.TemporaryDirectory() as td:
            root = _copy_minimal_install(Path(td) / "inst")
            tree_sha, _ = compute_code_tree_sha256(root)
            base_manifest = {
                "schema_version": 1,
                "version": "0.1.1",
                "release_tag": "v0.1.1",
                "runtime_abi_version": RUNTIME_ABI_VERSION,
                "git_commit": "abcdef1234567",
                "code_sha256": tree_sha,
                "overlay": {
                    "filename": "SlicerDSpace-v0.1.1-CodeOverlay.tar.gz",
                    "url": "http://127.0.0.1:1/SlicerDSpace-v0.1.1-CodeOverlay.tar.gz",
                    "sha256": "0" * 64,
                    "size_bytes": 123,
                },
            }
            self.assertEqual(
                check_for_code_update(module_root=root, manifest=base_manifest)["status"],
                "up_to_date",
            )
            newer = dict(base_manifest, code_sha256="1" * 64)
            self.assertEqual(
                check_for_code_update(module_root=root, manifest=newer)["status"],
                "update_available",
            )
            abi_bump = dict(newer, runtime_abi_version=RUNTIME_ABI_VERSION + 1)
            self.assertEqual(
                check_for_code_update(module_root=root, manifest=abi_bump)["status"],
                "runtime_upgrade_required",
            )

    def test_http_and_offline_overlay_apply_and_preserve_untracked_files(self) -> None:
        from shared.code_updater import RUNTIME_ABI_VERSION, compute_code_tree_sha256, download_and_apply_code_overlay

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            old_root = _copy_minimal_install(tmp / "installed")
            (old_root / "custom_user_note.txt").write_text("keep me", encoding="utf-8")
            stale_pyc = old_root / "shared" / "__pycache__" / "helper.cpython-312.pyc"
            stale_pyc.parent.mkdir(parents=True, exist_ok=True)
            stale_pyc.write_bytes(b"stale-pyc")

            new_tree = _copy_minimal_install(
                tmp / "new_src",
                extra_files={
                    "SlicerDSpace.py": b"# entry v2\nVALUE = 2\n",
                    "shared/helper.py": b"def version():\n    return 'v2'\n",
                    "shared/new_module.py": b"ADDED = True\n",
                },
            )
            (new_tree / "shared/obsolete.py").unlink()
            new_sha, new_files = compute_code_tree_sha256(new_tree)
            overlay_bytes_map = {rel: (new_tree / rel).read_bytes() for rel in new_files}
            overlay_tgz = tmp / "server" / "SlicerDSpace-v0.1.2-CodeOverlay.tar.gz"
            size_bytes, tar_sha = _build_tar_gz(overlay_tgz, overlay_bytes_map)

            class Handler(http.server.BaseHTTPRequestHandler):
                def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                    pass

                def do_GET(self) -> None:  # noqa: N802
                    if self.path.endswith("/update_manifest.json"):
                        body = json.dumps(manifest).encode("utf-8")
                    elif self.path.endswith("/SlicerDSpace-v0.1.2-CodeOverlay.tar.gz"):
                        body = overlay_tgz.read_bytes()
                    else:
                        self.send_response(404)
                        self.end_headers()
                        return
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

            srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            port = srv.server_address[1]
            manifest = {
                "schema_version": 1,
                "version": "0.1.2",
                "release_tag": "v0.1.2",
                "runtime_abi_version": RUNTIME_ABI_VERSION,
                "git_commit": "1234567890abcdef",
                "summary": "Fast patch",
                "overlay": {
                    "filename": overlay_tgz.name,
                    "url": f"http://127.0.0.1:{port}/{overlay_tgz.name}",
                    "sha256": tar_sha,
                    "size_bytes": size_bytes,
                },
                "code_sha256": new_sha,
            }
            t = threading.Thread(target=srv.serve_forever, daemon=True)
            t.start()
            progress_events: list[tuple[int, int]] = []
            try:
                res = download_and_apply_code_overlay(
                    module_root=old_root,
                    manifest_url=f"http://127.0.0.1:{port}/update_manifest.json",
                    progress_callback=lambda d, tot: progress_events.append((d, tot)),
                )
            finally:
                srv.shutdown()
                srv.server_close()

            self.assertEqual(res["status"], "applied")
            self.assertTrue(res["restart_required"])
            self.assertEqual(res["removed_files"], 1)
            self.assertGreater(len(progress_events), 0)
            self.assertEqual(compute_code_tree_sha256(old_root)[0], new_sha)
            self.assertIn("VALUE = 2", (old_root / "SlicerDSpace.py").read_text(encoding="utf-8"))
            self.assertTrue((old_root / "shared/new_module.py").is_file())
            self.assertFalse((old_root / "shared/obsolete.py").exists())
            self.assertFalse(stale_pyc.exists())
            self.assertEqual((old_root / "custom_user_note.txt").read_text(encoding="utf-8"), "keep me")

            # Now test offline folder apply from fresh old install.
            old2 = _copy_minimal_install(tmp / "installed_offline")
            offline_dir = tmp / "offline_pack_dir"
            offline_dir.mkdir()
            shutil.copy2(overlay_tgz, offline_dir / overlay_tgz.name)
            (offline_dir / "update_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            res2 = download_and_apply_code_overlay(module_root=old2, offline_dir=offline_dir)
            self.assertEqual(res2["status"], "applied")
            self.assertEqual(compute_code_tree_sha256(old2)[0], new_sha)

    def test_cli_worker_check_apply_and_error_json(self) -> None:
        from shared.code_updater import RUNTIME_ABI_VERSION, compute_code_tree_sha256

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            old_root = _copy_minimal_install(tmp / "installed")
            new_tree = _copy_minimal_install(
                tmp / "new_src",
                extra_files={
                    "SlicerDSpace.py": b"# entry v2\nVALUE = 2\n",
                    "shared/helper.py": b"def version():\n    return 'v2'\n",
                },
            )
            new_sha, new_files = compute_code_tree_sha256(new_tree)
            overlay_bytes_map = {rel: (new_tree / rel).read_bytes() for rel in new_files}
            overlay_tgz = tmp / "srv" / "SlicerDSpace-v0.1.2-CodeOverlay.tar.gz"
            size_bytes, tar_sha = _build_tar_gz(overlay_tgz, overlay_bytes_map)

            class Handler(http.server.BaseHTTPRequestHandler):
                def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                    pass

                def do_GET(self) -> None:  # noqa: N802
                    if self.path.endswith("/update_manifest.json"):
                        body = json.dumps(manifest).encode("utf-8")
                    elif self.path.endswith(f"/{overlay_tgz.name}"):
                        body = overlay_tgz.read_bytes()
                    else:
                        self.send_response(404)
                        self.end_headers()
                        return
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

            srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            port = srv.server_address[1]
            manifest = {
                "schema_version": 1,
                "version": "0.1.2",
                "release_tag": "v0.1.2",
                "runtime_abi_version": RUNTIME_ABI_VERSION,
                "git_commit": "abcdef1234567",
                "summary": "CLI worker test",
                "overlay": {
                    "filename": overlay_tgz.name,
                    "url": f"http://127.0.0.1:{port}/{overlay_tgz.name}",
                    "sha256": tar_sha,
                    "size_bytes": size_bytes,
                },
                "code_sha256": new_sha,
            }
            worker_script = str(REPO_ROOT / "shared" / "code_updater.py")
            chk_res = tmp / "chk_result.json"
            app_res = tmp / "app_result.json"
            err_res = tmp / "err_result.json"
            man_file = tmp / "manifest.json"
            clean_env = dict(os.environ)
            clean_env.pop("SLICERDSPACE_OFFLINE_DIR", None)

            t = threading.Thread(target=srv.serve_forever, daemon=True)
            t.start()
            try:
                chk_proc = subprocess.run(
                    [
                        sys.executable,
                        "-u",
                        worker_script,
                        "check",
                        "--module-root",
                        str(old_root),
                        "--result-file",
                        str(chk_res),
                        "--manifest-url",
                        f"http://127.0.0.1:{port}/update_manifest.json",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    env=clean_env,
                )
                self.assertEqual(chk_proc.returncode, 0, chk_proc.stdout + chk_proc.stderr)
                self.assertIn("SLICERDSPACE_UPDATE_STAGE:checking", chk_proc.stdout)
                chk_payload = json.loads(chk_res.read_text(encoding="utf-8"))
                self.assertTrue(chk_payload["ok"])
                self.assertEqual(chk_payload["result"]["status"], "update_available")

                man_file.write_text(
                    json.dumps(chk_payload["result"]["remote"]),
                    encoding="utf-8",
                )
                app_proc = subprocess.run(
                    [
                        sys.executable,
                        "-u",
                        worker_script,
                        "apply",
                        "--module-root",
                        str(old_root),
                        "--manifest-file",
                        str(man_file),
                        "--result-file",
                        str(app_res),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    env=clean_env,
                )
                self.assertEqual(app_proc.returncode, 0, app_proc.stdout + app_proc.stderr)
                self.assertIn("SLICERDSPACE_UPDATE_PROGRESS:", app_proc.stdout)
                app_payload = json.loads(app_res.read_text(encoding="utf-8"))
                self.assertTrue(app_payload["ok"])
                self.assertEqual(app_payload["result"]["status"], "applied")
                self.assertEqual(compute_code_tree_sha256(old_root)[0], new_sha)
            finally:
                srv.shutdown()
                srv.server_close()

            err_proc = subprocess.run(
                [
                    sys.executable,
                    "-u",
                    worker_script,
                    "check",
                    "--module-root",
                    str(old_root),
                    "--result-file",
                    str(err_res),
                    "--manifest-url",
                    f"http://127.0.0.1:{port}/update_manifest.json",
                    "--timeout",
                    "2",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                env=clean_env,
            )
            self.assertNotEqual(err_proc.returncode, 0)
            err_payload = json.loads(err_res.read_text(encoding="utf-8"))
            self.assertFalse(err_payload["ok"])
            self.assertTrue(err_payload["error"])

    def test_security_rejections_and_rollback(self) -> None:
        from shared.code_updater import RUNTIME_ABI_VERSION, compute_code_tree_sha256, download_and_apply_code_overlay

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            inst = _copy_minimal_install(tmp / "inst")
            orig_sha, _ = compute_code_tree_sha256(inst)

            new_tree = _copy_minimal_install(
                tmp / "v2",
                extra_files={"shared/helper.py": b"def version():\n    return 'v2'\n"},
            )
            v2_sha, v2_files = compute_code_tree_sha256(new_tree)
            v2_map = {rel: (new_tree / rel).read_bytes() for rel in v2_files}

            # 1. Symlink / traversal archive rejection.
            bad_sym = tmp / "bad_sym.tar.gz"
            sz_sym, sha_sym = _build_tar_gz(bad_sym, v2_map, symlink_member="shared/evil_link")
            offline = tmp / "off"
            offline.mkdir()
            shutil.copy2(bad_sym, offline / bad_sym.name)
            bad_manifest = {
                "schema_version": 1,
                "version": "0.1.2",
                "runtime_abi_version": RUNTIME_ABI_VERSION,
                "code_sha256": v2_sha,
                "overlay": {
                    "filename": bad_sym.name,
                    "url": "http://127.0.0.1:1/bad.tar.gz",
                    "sha256": sha_sym,
                    "size_bytes": sz_sym,
                },
            }
            with self.assertRaises(ValueError):
                download_and_apply_code_overlay(module_root=inst, manifest=bad_manifest, offline_dir=offline)
            self.assertEqual(compute_code_tree_sha256(inst)[0], orig_sha)

            bad_trav = tmp / "bad_trav.tar.gz"
            sz_t, sha_t = _build_tar_gz(bad_trav, v2_map, traversal_member="../escape.txt")
            shutil.copy2(bad_trav, offline / bad_trav.name)
            with self.assertRaises(ValueError):
                download_and_apply_code_overlay(
                    module_root=inst,
                    manifest=dict(bad_manifest, overlay={**bad_manifest["overlay"], "filename": bad_trav.name, "sha256": sha_t, "size_bytes": sz_t}),
                    offline_dir=offline,
                )

            # 2. Git checkout protection.
            (inst / ".git").mkdir()
            good_tgz = tmp / "good.tar.gz"
            sz_g, sha_g = _build_tar_gz(good_tgz, v2_map)
            shutil.copy2(good_tgz, offline / good_tgz.name)
            good_manifest = dict(
                bad_manifest,
                overlay={**bad_manifest["overlay"], "filename": good_tgz.name, "sha256": sha_g, "size_bytes": sz_g},
            )
            with self.assertRaises(RuntimeError):
                download_and_apply_code_overlay(module_root=inst, manifest=good_manifest, offline_dir=offline)
            shutil.rmtree(inst / ".git")

            # 3. Read-only directory rejection.
            with patch("shared.code_updater._is_directory_writable", return_value=False):
                with self.assertRaises(PermissionError):
                    download_and_apply_code_overlay(module_root=inst, manifest=good_manifest, offline_dir=offline)

            # 3b. SHA-256 mismatch and ABI bump rejection.
            bad_sha_manifest = dict(
                bad_manifest,
                overlay={**good_manifest["overlay"], "sha256": "0" * 64},
            )
            with self.assertRaises(RuntimeError):
                download_and_apply_code_overlay(module_root=inst, manifest=bad_sha_manifest, offline_dir=offline)

            abi_manifest = dict(good_manifest, runtime_abi_version=RUNTIME_ABI_VERSION + 1)
            with self.assertRaises(RuntimeError):
                download_and_apply_code_overlay(module_root=inst, manifest=abi_manifest, offline_dir=offline)

            # 4. Mid-apply rollback restores original files atomically.
            def _fail_second_file(rel: str) -> None:
                if rel == "shared/helper.py":
                    raise OSError("simulated disk write failure")

            with self.assertRaises(OSError):
                download_and_apply_code_overlay(
                    module_root=inst,
                    manifest=good_manifest,
                    offline_dir=offline,
                    _apply_hook=_fail_second_file,
                )
            self.assertEqual(compute_code_tree_sha256(inst)[0], orig_sha)
            self.assertIn("VALUE = 1", (inst / "SlicerDSpace.py").read_text(encoding="utf-8"))

    def test_ui_wiring_and_styles_below_log(self) -> None:
        compact_src = (REPO_ROOT / "shared/ui_compact.py").read_text(encoding="utf-8")
        tree = ast.parse(compact_src)
        install_fn = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "install_compact_layout"
        )
        calls = [n for n in ast.walk(install_fn) if isinstance(n, ast.Call)]
        upd_line = next(n.lineno for n in calls if isinstance(n.func, ast.Name) and n.func.id == "install_update_panel")
        stretch_line = max(
            n.lineno
            for n in calls
            if isinstance(n.func, ast.Attribute)
            and n.func.attr == "addStretch"
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "layout"
        )
        self.assertLess(upd_line, stretch_line)

        lang_fn = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "apply_compact_language"
        )
        lang_calls = [n for n in ast.walk(lang_fn) if isinstance(n, ast.Call)]
        self.assertTrue(
            any(isinstance(n.func, ast.Name) and n.func.id == "refresh_update_panel_language" for n in lang_calls)
        )

        widget_src = (REPO_ROOT / "shared/module_widget.py").read_text(encoding="utf-8")
        self.assertIn("cleanup_update_panel", widget_src)

        styles_src = (REPO_ROOT / "shared/ui_styles.py").read_text(encoding="utf-8")
        self.assertEqual(styles_src.count("QPushButton#checkUpdatesButton {"), 2)
        self.assertEqual(styles_src.count("QFrame#codeUpdateBarFrame {"), 2)
        self.assertEqual(styles_src.count("QLabel#codeUpdateStatusLabel {"), 2)

        update_panel_src = (REPO_ROOT / "shared/ui_update_panel.py").read_text(encoding="utf-8")
        self.assertIn("layout.indexOf(log_box)", update_panel_src)
        self.assertIn("QProcess", update_panel_src)
        self.assertNotIn("import threading", update_panel_src)
        self.assertNotIn("threading.Thread", update_panel_src)


if __name__ == "__main__":
    unittest.main()
