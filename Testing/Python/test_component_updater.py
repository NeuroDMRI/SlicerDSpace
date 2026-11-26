"""Unit tests for unified differential component updates (models, sources, runtimes)."""

from __future__ import annotations

import hashlib
import http.server
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import threading
import unittest
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_tar_gz(dest: Path, members: dict[str, bytes]) -> tuple[int, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dest, "w:gz", format=tarfile.PAX_FORMAT) as tf:
        for rel, raw in sorted(members.items()):
            info = tarfile.TarInfo(name=rel)
            info.size = len(raw)
            info.mode = 0o755 if ("/bin/" in rel or rel.startswith("bin/")) else 0o644
            tf.addfile(info, io.BytesIO(raw))
    raw_arc = dest.read_bytes()
    return len(raw_arc), _sha256_bytes(raw_arc)


class ComponentUpdaterTests(unittest.TestCase):
    def test_manifest_attaches_components_and_keeps_v011_schema_v1_compat(self) -> None:
        from shared.code_updater import _validate_manifest_payload
        from shared.component_updater import build_components_manifest
        from test_code_updater import _load_repackage_module

        comp = build_components_manifest(REPO_ROOT)
        self.assertIn("ddsurfer", comp["model_packs"])
        self.assertIn("ddseg", comp["model_packs"])
        self.assertIn("wma_org", comp["wma_packs"])
        self.assertIn("wma", comp["algorithm_sources"])
        self.assertTrue(comp["runtimes"]["surface"])
        self.assertEqual(len(comp["runtimes"]["ukf_standalone"]), 3)

        mod = _load_repackage_module()
        with tempfile.TemporaryDirectory() as td:
            manifest, _ = mod.build_code_overlay(Path(td), release_tag="v1.0.0")
            self.assertEqual(manifest["schema_version"], 1)
            self.assertIn("components", manifest)
            # Must pass v0.1.1's strict schema_version == 1 validator cleanly.
            validated = _validate_manifest_payload(manifest)
            self.assertEqual(validated["schema_version"], 1)
            self.assertIn("components", validated)

    def test_per_file_model_scan_replaces_only_changed_weights_in_place_and_readonly_fallback(self) -> None:
        from shared.component_updater import apply_component_updates, scan_component_updates

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            ext_root = tmp / "SlicerDSpace"
            mod_root = ext_root / "lib" / "Slicer-5.10" / "qt-scripted-modules"
            mod_root.mkdir(parents=True)
            bundled_models = ext_root / "bundled-model-assets"
            user_models = tmp / "user-models"

            unchanged_bytes = b"template-unchanged-v1"
            old_weight_bytes = b"ddsurfer-old-v011-weight"
            new_weight_bytes = b"ddsurfer-new-v100-weight-updated"
            ddseg_bytes = b"ddseg-unchanged-weight"

            (bundled_models / "surface" / "template").mkdir(parents=True)
            (bundled_models / "surface" / "weights").mkdir(parents=True)
            (bundled_models / "segmentation" / "weights").mkdir(parents=True)
            (bundled_models / "surface" / "template" / "lh.obj").write_bytes(unchanged_bytes)
            (bundled_models / "surface" / "weights" / "lh_wm.pt").write_bytes(old_weight_bytes)
            (bundled_models / "segmentation" / "weights" / "axial.pt").write_bytes(ddseg_bytes)

            offline_dir = tmp / "offline"
            offline_dir.mkdir()
            new_sha = _sha256_bytes(new_weight_bytes)
            (offline_dir / f"{new_sha}-lh_wm.pt").write_bytes(new_weight_bytes)

            components = {
                "version": 1,
                "model_packs": {
                    "ddsurfer": {
                        "id": "ddsurfer",
                        "files": {
                            "surface/template/lh.obj": {
                                "sha256": _sha256_bytes(unchanged_bytes),
                                "size_bytes": len(unchanged_bytes),
                                "urls": ["http://127.0.0.1:9/lh.obj"],
                            },
                            "surface/weights/lh_wm.pt": {
                                "sha256": new_sha,
                                "size_bytes": len(new_weight_bytes),
                                "urls": [f"https://github.com/NeuroDMRI/SlicerDSpace/releases/download/assets/{new_sha}-lh_wm.pt"],
                            },
                        },
                    },
                    "ddseg": {
                        "id": "ddseg",
                        "files": {
                            "segmentation/weights/axial.pt": {
                                "sha256": _sha256_bytes(ddseg_bytes),
                                "size_bytes": len(ddseg_bytes),
                                "urls": ["http://127.0.0.1:9/axial.pt"],
                            }
                        },
                    },
                    "ddparcel": {
                        "id": "ddparcel",
                        "files": {
                            "parcellation/weights/not_installed.pkl": {
                                "sha256": "a" * 64,
                                "size_bytes": 999,
                                "urls": ["http://127.0.0.1:9/not_installed.pkl"],
                            }
                        },
                    },
                },
                "wma_packs": {
                    "wma_org": {
                        "id": "wma_org",
                        "directory": "bundles/wma",
                        "index_relpath": "bundles/wma/extracted_resource_index.json",
                        "index_sha256": "b" * 64,
                        "filename": "SlicerDSpace-WMA-ORG-v1.tar.gz",
                        "url": "http://127.0.0.1:9/org.tar.gz",
                        "sha256": "c" * 64,
                        "size_bytes": 1700000000,
                        "parts": [],
                    }
                },
                "algorithm_sources": {},
                "runtimes": {},
            }

            with patch.dict(os.environ, {"SLICERDSPACE_MODEL_ASSETS_DIR": str(user_models)}):
                pending = scan_component_updates(mod_root, components)
                # Only the single changed DDSurfer checkpoint is queued; unchanged DDSeg, unchanged template,
                # uninstalled DDParcel, and uninstalled WMA ORG are all skipped.
                self.assertEqual([u["id"] for u in pending], ["model:surface/weights/lh_wm.pt"])

                res = apply_component_updates(mod_root, pending, offline_dir=offline_dir)
                self.assertEqual(res["updated_count"], 1)
                self.assertEqual((bundled_models / "surface" / "weights" / "lh_wm.pt").read_bytes(), new_weight_bytes)
                self.assertFalse((user_models / "surface" / "weights" / "lh_wm.pt").exists())
                self.assertEqual(scan_component_updates(mod_root, components), [])

                # Now simulate a read-only bundled-model-assets directory: fallback writes to user_models.
                (bundled_models / "surface" / "weights" / "lh_wm.pt").write_bytes(old_weight_bytes)
                pending_ro = scan_component_updates(mod_root, components)
                self.assertEqual(len(pending_ro), 1)
                with patch("shared.component_updater._ensure_writable_or_parent", return_value=False):
                    apply_component_updates(mod_root, pending_ro, offline_dir=offline_dir)
                self.assertEqual((user_models / "surface" / "weights" / "lh_wm.pt").read_bytes(), new_weight_bytes)
                self.assertEqual(scan_component_updates(mod_root, components), [])

    def test_v011_companion_sources_and_runtimes_upgrade_and_unified_apply(self) -> None:
        from shared.code_updater import (
            RUNTIME_ABI_VERSION,
            check_for_updates,
            check_local_component_updates,
            compute_code_tree_sha256,
            download_and_apply_updates,
        )
        from test_code_updater import _build_tar_gz, _copy_minimal_install

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            ext_root = tmp / "SlicerDSpace"
            mod_root = _copy_minimal_install(ext_root / "lib" / "Slicer-5.10" / "qt-scripted-modules")
            bundled_src = ext_root / "bundled-sources"
            bundled_rt = ext_root / "bundled-runtimes"
            bundled_rt.mkdir(parents=True)
            user_models = tmp / "user-models"
            user_src = tmp / "user-src"

            # Legacy WMA source folder (like v0.1.1 wma-4532b911386d)
            old_wma = bundled_src / "wma-4532b911386d"
            (old_wma / "bin").mkdir(parents=True)
            (old_wma / "bin" / "wm_apply_ORG_atlas_to_subject.sh").write_text("#!/bin/sh\necho old\n", encoding="utf-8")

            # Build fake new WMA source archive, new UKF archive, and new Surface runtime archive
            wma_rev = "93a7a7cf00be45c93009b5247af59a9314e31a82"
            wma_arc = tmp / "srv" / "wma.tar.gz"
            wma_sz, wma_sha = _write_tar_gz(
                wma_arc,
                {
                    f"whitematteranalysis_gpu-{wma_rev}/bin/wm_apply_ORG_atlas_to_subject.sh": b"#!/bin/sh\necho new\n",
                    f"whitematteranalysis_gpu-{wma_rev}/License.txt": b"BSD",
                },
            )

            import platform as _plat
            from shared.component_updater import _host_platform_pair

            sys_name, arch = _host_platform_pair()
            ukf_dir = f"ukf-{sys_name.lower()}-{arch}"
            ukf_exe = "bin/UKFTractography.exe" if sys_name == "Windows" else "bin/UKFTractography"
            ukf_rt_bytes = json.dumps(
                {
                    "kind": "ukf-standalone",
                    "platform": f"{sys_name.lower()}-{arch}",
                    "arch": arch,
                    "executable": ukf_exe,
                    "source_commit": "newcommit123",
                }
            ).encode("utf-8")
            ukf_arc = tmp / "srv" / "ukf-standalone.tar.gz"
            ukf_sz, ukf_sha = _write_tar_gz(
                ukf_arc,
                {
                    f"{ukf_dir}/{ukf_exe}": b"#!/bin/sh\necho UKF\n",
                    f"{ukf_dir}/runtime.json": ukf_rt_bytes,
                },
            )

            # Outdated surface runtime currently installed
            old_surf = bundled_rt / f"surface-runtime-{sys_name.lower()}-{arch}"
            (old_surf / "bin").mkdir(parents=True)
            (old_surf / "bin" / "mris_inflate").write_bytes(b"old")
            (old_surf / "runtime.json").write_text(
                json.dumps(
                    {
                        "activation_key_included": False,
                        "recipe_sha256": "old_recipe",
                        "files": {"bin/mris_inflate": {"bytes": 3, "sha256": _sha256_bytes(b"old")}},
                    }
                ),
                encoding="utf-8",
            )

            new_surf_bin = b"#!/bin/sh\necho mris_inflate_v2\n"
            new_surf_rt_json = json.dumps(
                {
                    "activation_key_included": False,
                    "recipe_sha256": "recipe_v100",
                    "files": {"bin/mris_inflate": {"bytes": len(new_surf_bin), "sha256": _sha256_bytes(new_surf_bin)}},
                }
            ).encode("utf-8")
            surf_arc = tmp / "srv" / "surface-runtime.tar.gz"
            surf_sz, surf_sha = _write_tar_gz(
                surf_arc,
                {
                    "bin/mris_inflate": new_surf_bin,
                    "runtime.json": new_surf_rt_json,
                },
            )

            # New code overlay v2
            v2_tree = _copy_minimal_install(
                tmp / "v2_src",
                extra_files={"SlicerDSpace.py": b"# entry v2\nVALUE = 2\n"},
            )
            v2_sha, v2_files = compute_code_tree_sha256(v2_tree)
            ov_arc = tmp / "srv" / "SlicerDSpace-v1.0.0-CodeOverlay.tar.gz"
            ov_sz, ov_sha = _build_tar_gz(ov_arc, {rel: (v2_tree / rel).read_bytes() for rel in v2_files})

            srv_files = {
                f"/{ov_arc.name}": ov_arc,
                "/wma.tar.gz": wma_arc,
                "/ukf-standalone.tar.gz": ukf_arc,
                "/surface-runtime.tar.gz": surf_arc,
            }

            class Handler(http.server.BaseHTTPRequestHandler):
                def log_message(self, fmt: str, *args: object) -> None:
                    pass

                def do_GET(self) -> None:  # noqa: N802
                    if self.path in srv_files:
                        body = srv_files[self.path].read_bytes()
                        self.send_response(200)
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                        return
                    self.send_response(404)
                    self.end_headers()

            srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            port = srv.server_address[1]
            t = threading.Thread(target=srv.serve_forever, daemon=True)
            t.start()

            components = {
                "version": 1,
                "model_packs": {},
                "wma_packs": {},
                "algorithm_sources": {
                    "wma": {
                        "name": "wma",
                        "repository": "NeuroDMRI/whitematteranalysis_gpu",
                        "revision": wma_rev,
                        "sha256": wma_sha,
                        "size_bytes": wma_sz,
                        "entry": "bin/wm_apply_ORG_atlas_to_subject.sh",
                        "paths": ["bin", "License.txt"],
                        "url": f"http://127.0.0.1:{port}/wma.tar.gz",
                    }
                },
                "runtimes": {
                    "surface": [
                        {
                            "system": sys_name,
                            "machine": arch,
                            "recipe_sha256": "recipe_v100",
                            "sha256": surf_sha,
                            "bytes": surf_sz,
                            "unpacked_bytes": len(new_surf_bin),
                            "published": True,
                            "url": f"http://127.0.0.1:{port}/surface-runtime.tar.gz",
                        }
                    ],
                    "ukf_standalone": [
                        {
                            "system": sys_name,
                            "machine": arch,
                            "dir_name": ukf_dir,
                            "filename": "ukf-standalone.tar.gz",
                            "format": "tar.gz",
                            "bytes": ukf_sz,
                            "sha256": ukf_sha,
                            "executable": ukf_exe,
                            "source_commit": "newcommit123",
                            "requires_avx2": False,
                            "url": f"http://127.0.0.1:{port}/ukf-standalone.tar.gz",
                        }
                    ],
                    "native": [],
                },
            }
            manifest = {
                "schema_version": 1,
                "version": "1.0.0",
                "release_tag": "v1.0.0",
                "runtime_abi_version": RUNTIME_ABI_VERSION,
                "git_commit": "abcdef1234567",
                "code_sha256": v2_sha,
                "overlay": {
                    "filename": ov_arc.name,
                    "url": f"http://127.0.0.1:{port}/{ov_arc.name}",
                    "sha256": ov_sha,
                    "size_bytes": ov_sz,
                },
                "components": components,
            }

            try:
                with patch.dict(os.environ, {"SLICERDSPACE_MODEL_ASSETS_DIR": str(user_models)}), patch(
                    "shared.backend_sources.source_root", return_value=user_src
                ):
                    chk = check_for_updates(module_root=mod_root, manifest=manifest)
                    self.assertEqual(chk["status"], "update_available")
                    self.assertTrue(chk["code_update_available"])
                    self.assertEqual(
                        sorted(u["id"] for u in chk["component_updates"]),
                        ["runtime:surface", "runtime:ukf", "source:wma"],
                    )
                    self.assertEqual(chk["total_size_bytes"], ov_sz + wma_sz + ukf_sz + surf_sz)

                    prog: list[tuple[int, int]] = []
                    applied = download_and_apply_updates(
                        module_root=mod_root,
                        manifest=manifest,
                        progress_callback=lambda d, tot: prog.append((d, tot)),
                    )
                    self.assertEqual(applied["status"], "applied")
                    self.assertTrue(applied["restart_required"])
                    self.assertEqual(applied["updated_component_count"], 3)
                    self.assertTrue(prog)
                    # Old WMA source sibling removed, new WMA source verified in place
                    self.assertFalse(old_wma.exists())
                    self.assertTrue((bundled_src / f"wma-{wma_rev[:12]}" / ".slicerdspace-source.json").is_file())
                    # UKF and Surface runtime updated in place
                    self.assertTrue((bundled_rt / ukf_dir / ukf_exe).is_file())
                    self.assertEqual((old_surf / "bin" / "mris_inflate").read_bytes(), new_surf_bin)

                    # Re-check now reports completely up_to_date (0 bytes)!
                    chk2 = check_for_updates(module_root=mod_root, manifest=manifest)
                    self.assertEqual(chk2["status"], "up_to_date")
                    self.assertEqual(chk2["total_size_bytes"], 0)

                    # Simulate components-only update (e.g. post-v0.1.1 restart handoff: code already matches!)
                    (old_surf / ".slicerdspace_archive_sha256").write_text("0" * 64 + "\n", encoding="utf-8")
                    with patch("shared.component_updater.build_components_manifest", return_value=components):
                        loc_chk = check_local_component_updates(module_root=mod_root)
                        self.assertEqual(loc_chk["status"], "update_available")
                        self.assertTrue(loc_chk["components_only"])
                        comp_only_res = download_and_apply_updates(module_root=mod_root, components_only=True)
                        self.assertEqual(comp_only_res["status"], "applied")
                        self.assertFalse(comp_only_res["restart_required"])
                        self.assertEqual(comp_only_res["updated_components"], ["runtime:surface"])
            finally:
                srv.shutdown()
                srv.server_close()


if __name__ == "__main__":
    unittest.main()
