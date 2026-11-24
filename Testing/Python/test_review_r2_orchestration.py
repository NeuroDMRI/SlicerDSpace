"""Round-2 code-review regression tests for Group A (Items 1, 2, 6, 7).

Covers:
- Item 1 (P1): SIFT2 weights require verified CSD/ACT manifest+QC provenance and matching TCK content.
- Item 2 (P1): Canonical workspace lock across ``<W>`` / ``<W>/outputs`` aliases & locked failure writes.
- Item 6 (P1): Archived run QC snapshots and output fingerprint validation preventing stale/new-run bleed.
- Item 7 (P1): Update rollback verifies every delete/restore and preserves journal/backups on Windows locks.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

import nibabel as nib
from nibabel.streamlines import ArraySequence, TckFile, Tractogram
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _write_minimal_raw_subject(root: Path) -> Path:
    inp = root / "input"
    inp.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.ones((6, 6, 6, 4), dtype=np.float32), np.eye(4)), str(inp / "data.nii.gz"))
    nib.save(nib.Nifti1Image(np.ones((6, 6, 6), dtype=np.uint8), np.eye(4)), str(inp / "nodif_brain_mask.nii.gz"))
    (inp / "bvals").write_text("0 1000 1000 1000\n", encoding="utf-8")
    (inp / "bvecs").write_text("0 1 0 0\n0 0 1 0\n0 0 0 1\n", encoding="utf-8")
    return root


def _write_two_edge_parcellation(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.zeros((6, 6, 6), dtype=np.int16)
    labels[1, 1, 1] = 10
    labels[4, 1, 1] = 20
    labels[1, 4, 1] = 30
    labels[4, 4, 1] = 40
    nib.save(nib.Nifti1Image(labels, np.eye(4)), str(path))
    return path


def _write_two_streamline_tck(path: Path, *, swapped: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    s_a = np.asarray([[1.0, 1.0, 1.0], [4.0, 1.0, 1.0]], dtype=np.float32)  # connects 10 <-> 20
    s_b = np.asarray([[1.0, 4.0, 1.0], [4.0, 4.0, 1.0]], dtype=np.float32)  # connects 30 <-> 40
    seq = ArraySequence()
    for line in ((s_b, s_a) if swapped else (s_a, s_b)):
        seq.append(line)
    nib.streamlines.save(TckFile(Tractogram(seq, affine_to_rasmm=np.eye(4))), path.as_posix())
    return path


def _parse_connectome_edges(csv_path: Path) -> dict[tuple[int, int], float]:
    edges: dict[tuple[int, int], float] = {}
    for line in csv_path.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip() or line.startswith("#"):
            continue
        src, dst, val = line.split(",")[:3]
        pair = (min(int(src), int(dst)), max(int(src), int(dst)))
        edges[pair] = float(val)
    return edges


class Sift2ProvenanceVerificationTests(unittest.TestCase):
    """[Item 1] SIFT2 weights must never pair with an unverified or swapped TCK when manifest/QC are missing."""

    def test_missing_manifest_and_qc_blocks_sift2_and_swapped_tck_invalidates_weights(self) -> None:
        from shared.io_layout import subject_layout
        from shared.orchestrator import (
            ProductOrchestratorConfig,
            _run_stage_dirs,
            _sift2_weights_current,
            plan_subject_products,
            run_subject_products,
        )
        from tractography.csd import _file_info as _file_record

        with tempfile.TemporaryDirectory() as td:
            root = _write_minimal_raw_subject(Path(td) / "sub01")
            layout = subject_layout(root)
            parcel = _write_two_edge_parcellation(layout.anatomy_input_dir("ddparcel") / "sub01-DDSurfer-wmparc.nii.gz")

            dirs = _run_stage_dirs(layout, "run1", root)
            dirs["tractogram_dir"].mkdir(parents=True, exist_ok=True)
            dirs["csd_dir"].mkdir(parents=True, exist_ok=True)
            tck = _write_two_streamline_tck(dirs["tractogram_dir"] / "sub01_csd_act_ifod2.tck", swapped=False)
            weights = dirs["tractogram_dir"] / "sub01_csd_act_ifod2_sift2_weights.txt"
            weights.write_text("2.0\n9.0\n", encoding="utf-8")
            csd_manifest = dirs["csd_dir"] / "csd_act_manifest.json"
            csd_qc = dirs["csd_dir"] / "csd_act_qc.json"

            # 1. Both CSD manifest AND QC missing: orphan weights MUST be rejected!
            self.assertFalse(_sift2_weights_current(weights, tck, csd_manifest, csd_qc, use_sift2=True))

            cfg_sift2_only = ProductOrchestratorConfig(
                subject_id="sub01",
                subject_root=root,
                targets=("connectome",),
                run_name="run1",
                tractography_backend="csd_act",
                connectome_backend="python",
                connectome_metrics="sift2",
                closure_applied=True,
                execute=True,
                resume=False,
            )
            with mock.patch("shared.orchestrator._resolve_parcellation", return_value=parcel):
                plan = plan_subject_products(cfg_sift2_only, write=False)
                conn_plan = next(item for item in plan.items if item.target == "connectome")
                self.assertEqual(conn_plan.status, "blocked")
                self.assertIn("provenance", conn_plan.reason.lower())

                res_orphan = run_subject_products(cfg_sift2_only)
            s2_orphan = next(t for t in res_orphan.tasks if t.name == "connectome.sift2_weighted")
            self.assertEqual(s2_orphan.status, "blocked")
            self.assertIn("provenance", s2_orphan.reason.lower())
            self.assertFalse((layout.connectome / "csd_act" / "sub01_sift2_weighted.csv").exists())

            # 2. Write valid matching CSD manifest & QC -> SIFT2 connectome runs and produces exact (2.0, 9.0).
            csd_manifest.write_text(
                json.dumps({
                    "status": "completed",
                    "parameters": {"use_sift2": True},
                    "outputs": {"tractogram": tck.as_posix(), "sift2_weights": weights.as_posix()},
                }),
                encoding="utf-8",
            )
            csd_qc.write_text(
                json.dumps({
                    "status": "completed",
                    "scientific_output_summary": {
                        "tck": {"file": _file_record(tck), "streamline_count": 2},
                        "sift2": {
                            "file": _file_record(weights),
                            "weight_count": 2,
                            "all_finite": True,
                            "all_nonnegative": True,
                        },
                    },
                }),
                encoding="utf-8",
            )
            self.assertTrue(_sift2_weights_current(weights, tck, csd_manifest, csd_qc, use_sift2=True))
            with mock.patch("shared.orchestrator._resolve_parcellation", return_value=parcel):
                res_valid = run_subject_products(cfg_sift2_only)
            s2_valid = next(t for t in res_valid.tasks if t.name == "connectome.sift2_weighted")
            self.assertEqual(s2_valid.status, "completed")
            edges_valid = _parse_connectome_edges(Path(s2_valid.outputs["matrix_path"]))
            self.assertEqual(edges_valid, {(10, 20): 2.0, (30, 40): 9.0})

            # 3. Swap streamline order in TCK (same streamline count=2, same byte size, even forged same mtime!).
            orig_stat = tck.stat()
            _write_two_streamline_tck(tck, swapped=True)
            os.utime(tck, ns=(orig_stat.st_atime_ns, orig_stat.st_mtime_ns))
            self.assertEqual(tck.stat().st_size, orig_stat.st_size)

            # Even with valid manifest/QC present, SHA-256 mismatch rejects swapped TCK immediately.
            self.assertFalse(_sift2_weights_current(weights, tck, csd_manifest, csd_qc, use_sift2=True))

            # Even after deleting BOTH manifest AND QC, swapped TCK + stale weights MUST remain rejected!
            csd_manifest.unlink()
            csd_qc.unlink()
            self.assertFalse(_sift2_weights_current(weights, tck, csd_manifest, csd_qc, use_sift2=True))

            with mock.patch("shared.orchestrator._resolve_parcellation", return_value=parcel):
                res_swapped = run_subject_products(cfg_sift2_only)
            s2_swapped = next(t for t in res_swapped.tasks if t.name == "connectome.sift2_weighted")
            self.assertEqual(s2_swapped.status, "blocked")
            # Published matrix must NOT have been silently overwritten with swapped (9.0, 2.0)!
            self.assertEqual(_parse_connectome_edges(Path(s2_valid.outputs["matrix_path"])), {(10, 20): 2.0, (30, 40): 9.0})

            # Explicit "sift2,fiber_count" still runs fiber_count while blocking unverified sift2_weighted.
            cfg_both = ProductOrchestratorConfig(
                subject_id="sub01",
                subject_root=root,
                targets=("connectome",),
                run_name="run1",
                tractography_backend="csd_act",
                connectome_backend="python",
                connectome_metrics="sift2,fiber_count",
                closure_applied=True,
                execute=True,
                resume=False,
            )
            with mock.patch("shared.orchestrator._resolve_parcellation", return_value=parcel):
                res_both = run_subject_products(cfg_both)
            by_name = {t.name: t for t in res_both.tasks}
            self.assertEqual(by_name["connectome.streamline_count"].status, "completed")
            self.assertEqual(by_name["connectome.sift2_weighted"].status, "blocked")


class WorkspaceLockAliasingAndFailureTests(unittest.TestCase):
    """[Item 2] Canonical lock path across <W> vs <W>/outputs aliases & no post-unlock 'failed' clobber."""

    def test_2a_workspace_lock_normalizes_w_vs_w_outputs_and_blocks_cross_process(self) -> None:
        from shared.io_layout import SubjectIOLayout, subject_layout

        with tempfile.TemporaryDirectory() as td:
            base = Path(td).resolve()
            src = _write_minimal_raw_subject(base / "source_sub")
            workspace = base / "workspace_W"
            workspace.mkdir()

            layouts = [
                subject_layout(src, output_root=workspace),
                subject_layout(src, output_root=workspace / "outputs"),
                subject_layout(src, output_root=workspace / "output"),
                subject_layout(src, output_root=workspace / "OUTPUTS"),
                subject_layout(workspace, output_root=None),
                SubjectIOLayout(src, output_root=workspace / "outputs"),
                SubjectIOLayout(src, output_root=workspace),
            ]
            # Also test symlink / Windows junction alias to <W>.
            alias_link = base / "workspace_alias"
            created_link = False
            try:
                if sys.platform == "win32":
                    proc = subprocess.run(
                        ["cmd", "/c", "mklink", "/J", str(alias_link), str(workspace)],
                        capture_output=True,
                    )
                    created_link = proc.returncode == 0
                else:
                    alias_link.symlink_to(workspace, target_is_directory=True)
                    created_link = True
            except OSError:
                created_link = False
            if created_link:
                layouts.append(subject_layout(src, output_root=alias_link))
                layouts.append(subject_layout(src, output_root=alias_link / "outputs"))

            expected_lock_dir = os.path.normcase(str((workspace / "cache").resolve()))
            for layout in layouts:
                self.assertEqual(
                    os.path.normcase(str(layout.workspace_lock_dir.resolve())),
                    expected_lock_dir,
                    f"Alias mismatch for layout={layout}",
                )

            # Real cross-process contention: Process A holds <W>, Process B tries <W>/outputs, Process C tries subject_root=<W>.
            ready_file = base / "holder.ready"
            stop_file = base / "holder.stop"
            holder_code = (
                "import sys, time\n"
                "from pathlib import Path\n"
                "from shared.io_layout import subject_layout\n"
                "from shared.workspace_lock import subject_workspace_lock\n"
                "src, out, ready, stop = map(Path, sys.argv[1:5])\n"
                "layout = subject_layout(src, output_root=out)\n"
                "with subject_workspace_lock(layout, subject_id='sub01', run_name='runW'):\n"
                "    ready.write_text('held', encoding='utf-8')\n"
                "    deadline = time.monotonic() + 25.0\n"
                "    while not stop.exists() and time.monotonic() < deadline:\n"
                "        time.sleep(0.05)\n"
            )
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), env.get("PYTHONPATH", "")])
            env.pop("SLICERDSPACE_WORKSPACE_LOCKS", None)
            holder = subprocess.Popen(
                [sys.executable, "-c", holder_code, str(src), str(workspace), str(ready_file), str(stop_file)],
                env=env,
            )
            try:
                deadline = time.monotonic() + 10.0
                while not ready_file.exists() and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertTrue(ready_file.exists(), "Holder process failed to acquire canonical workspace lock")

                contender_code = (
                    "import sys\n"
                    "from pathlib import Path\n"
                    "from shared.orchestrator import ProductOrchestratorConfig, run_subject_products\n"
                    "from shared.workspace_lock import WorkspaceLockError\n"
                    "src = Path(sys.argv[1])\n"
                    "out = Path(sys.argv[2]) if sys.argv[2] else None\n"
                    "cfg = ProductOrchestratorConfig(subject_id='sub01', subject_root=src, output_root=out, "
                    "targets=('validation',), run_name='runAlias', execute=True)\n"
                    "try:\n"
                    "    run_subject_products(cfg)\n"
                    "    print('UNEXPECTED_SUCCESS')\n"
                    "    sys.exit(0)\n"
                    "except WorkspaceLockError as exc:\n"
                    "    print('LOCKED:', exc)\n"
                    "    sys.exit(42)\n"
                )
                alias_cases = [
                    ("w_outputs", str(src), str(workspace / "outputs")),
                    ("w_output_upper", str(src), str(workspace / "OUTPUTS")),
                    ("subject_root_w", str(workspace), ""),
                ]
                if created_link:
                    alias_cases.append(("junction_outputs", str(src), str(alias_link / "outputs")))

                for label, s_arg, o_arg in alias_cases:
                    with self.subTest(alias=label):
                        proc = subprocess.run(
                            [sys.executable, "-c", contender_code, s_arg, o_arg],
                            capture_output=True,
                            text=True,
                            env=env,
                            timeout=20,
                        )
                        self.assertEqual(proc.returncode, 42, proc.stdout + proc.stderr)
                        self.assertIn(f"PID={holder.pid}", proc.stdout)
                        self.assertIn("run_name=runW", proc.stdout)
            finally:
                stop_file.write_text("stop", encoding="utf-8")
                holder.wait(timeout=10)
                if created_link and alias_link.exists():
                    try:
                        if sys.platform == "win32":
                            alias_link.rmdir()
                        else:
                            alias_link.unlink()
                    except OSError:
                        pass

    def test_2b_failed_manifest_written_strictly_while_holding_lock_and_never_clobbers_next_run(self) -> None:
        from shared.io_layout import subject_layout
        from shared.orchestrator import ProductOrchestratorConfig, _mark_orchestration_failed, run_subject_products
        from shared.workspace_lock import LOCK_FILENAME, WorkspaceLockError, is_workspace_lock_held, subject_workspace_lock

        with tempfile.TemporaryDirectory() as td:
            root = _write_minimal_raw_subject(Path(td) / "sub01")
            layout = subject_layout(root)
            manifest_path = layout.run_metadata / "run1_orchestration_manifest.json"
            lock_states_during_fail_write: list[tuple[bool, bool]] = []
            writing_started = threading.Event()
            allow_finish_write = threading.Event()

            orig_mark_failed = _mark_orchestration_failed

            def _instrumented_mark_failed(cfg, err, stage=""):
                writing_started.set()
                # Check lock is STILL held both in-process and on disk at the exact moment 'failed' is written!
                lock_states_during_fail_write.append(
                    (
                        is_workspace_lock_held(layout.workspace_lock_dir),
                        (layout.workspace_lock_dir / LOCK_FILENAME).is_file(),
                    )
                )
                allow_finish_write.wait(timeout=5.0)
                return orig_mark_failed(cfg, err, stage)

            task_b_acquired_before_fail_finished = False
            task_b_done = threading.Event()

            def _task_b_waiter():
                nonlocal task_b_acquired_before_fail_finished
                writing_started.wait(timeout=5.0)
                # While Task A is still inside _mark_orchestration_failed, Task B MUST NOT acquire the lock!
                try:
                    with subject_workspace_lock(layout, subject_id="sub01", run_name="run1"):
                        task_b_acquired_before_fail_finished = True
                except WorkspaceLockError:
                    pass
                allow_finish_write.set()
                # Now wait until Task A releases the lock, then Task B acquires it and writes 'running'.
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    try:
                        with subject_workspace_lock(layout, subject_id="sub01", run_name="run1"):
                            manifest_path.write_text(
                                json.dumps({"kind": "slicerddsurfer_product_orchestration", "version": 1, "status": "running", "owner": "taskB"}),
                                encoding="utf-8",
                            )
                            task_b_done.set()
                            return
                    except WorkspaceLockError:
                        time.sleep(0.02)

            t_b = threading.Thread(target=_task_b_waiter)
            t_b.start()
            cfg = ProductOrchestratorConfig(
                subject_id="sub01",
                subject_root=root,
                targets=("validation",),
                run_name="run1",
                execute=True,
            )
            try:
                with mock.patch("shared.orchestrator._mark_orchestration_failed", side_effect=_instrumented_mark_failed), mock.patch(
                    "shared.orchestrator._run_subject_products_locked",
                    side_effect=RuntimeError("boom inside locked stage"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "boom inside locked stage"):
                        run_subject_products(cfg)
            finally:
                allow_finish_write.set()
                t_b.join(timeout=10.0)

            self.assertEqual(lock_states_during_fail_write, [(True, True)])
            self.assertFalse(task_b_acquired_before_fail_finished)
            self.assertTrue(task_b_done.is_set())
            # Task B's freshly written 'running' status MUST survive intact!
            final_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(final_payload["status"], "running")
            self.assertEqual(final_payload["owner"], "taskB")

    def test_2b_pre_lock_validation_error_never_clobbers_active_owner_manifest(self) -> None:
        from shared.io_layout import subject_layout
        from shared.orchestrator import ProductOrchestratorConfig, run_subject_products
        from shared.workspace_lock import subject_workspace_lock

        with tempfile.TemporaryDirectory() as td:
            root = _write_minimal_raw_subject(Path(td) / "sub01")
            layout = subject_layout(root)
            layout.run_metadata.mkdir(parents=True, exist_ok=True)
            manifest_path = layout.run_metadata / "run1_orchestration_manifest.json"
            active_payload = {"kind": "slicerddsurfer_product_orchestration", "version": 1, "status": "running", "owner": "liveA"}
            manifest_path.write_text(json.dumps(active_payload), encoding="utf-8")

            ready = threading.Event()
            release = threading.Event()

            def _hold():
                with subject_workspace_lock(layout, subject_id="sub01", run_name="run1"):
                    ready.set()
                    release.wait(timeout=10.0)

            holder = threading.Thread(target=_hold)
            holder.start()
            self.assertTrue(ready.wait(timeout=5.0))
            try:
                # Raises ValueError inside _validate_bundle_tractography_source BEFORE early_lock!
                bad_cfg = ProductOrchestratorConfig(
                    subject_id="sub01",
                    subject_root=root,
                    targets=("bundles",),
                    run_name="run1",
                    tractography_backend="dti_det",
                    bundle_tractography_backend="invalid_backend_xyz",
                    execute=True,
                )
                with self.assertRaises(ValueError):
                    run_subject_products(bad_cfg)
                self.assertEqual(json.loads(manifest_path.read_text(encoding="utf-8")), active_payload)
            finally:
                release.set()
                holder.join(timeout=5.0)


class RunHistoryArchiveIntegrityTests(unittest.TestCase):
    """[Item 6] Archived runs preserve historical QC snapshots and never silently load newer overwritten products."""

    def test_6_rerun_same_name_snapshots_qc_and_marks_overwritten_outputs_superseded(self) -> None:
        from shared.auto_results import automatic_result_candidates
        from shared.io_layout import subject_layout
        from shared.orchestrator import _archive_existing_run_records
        from shared.result_scene import (
            build_manifest_artifact_fingerprints,
            manifest_output_status,
            recorded_qc_summary,
            result_candidates,
        )
        from tractography.wmql import WMQL_QC_VERSION

        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve() / "sub01"
            layout = subject_layout(root)
            layout.run_metadata.mkdir(parents=True, exist_ok=True)
            layout.diffusion_tensor.mkdir(parents=True, exist_ok=True)
            layout.multi_tensor_tractography.mkdir(parents=True, exist_ok=True)
            layout.query_bundle_extraction.mkdir(parents=True, exist_ok=True)
            dti_stage = layout.stage_cache("02_microstructure/dti")
            ukf_stage = layout.stage_cache("03_tractography/ukf")
            wmql_stage = layout.stage_cache("04_bundles/wmql")
            for folder in (dti_stage, ukf_stage, wmql_stage):
                folder.mkdir(parents=True, exist_ok=True)

            # Live files produced by Run 1 (completed, all QC passed)
            fa_live = layout.diffusion_tensor / "sub01_FA.nii.gz"
            ukf_live = layout.multi_tensor_tractography / "sub01_ukf.vtk"
            af_live = layout.query_bundle_extraction / "af.left.trk"
            fa_live.write_bytes(b"FA-RUN1-DATA-GOOD")
            ukf_live.write_bytes(b"UKF-RUN1-DATA-GOOD")
            af_live.write_bytes(b"WMQL-AF-RUN1-GOOD")

            dti_qc = dti_stage / "dti_qc.json"
            ukf_qc = ukf_stage / "ukf_qc.json"
            wmql_qc = wmql_stage / "wmql_qc.json"
            dti_qc.write_text(json.dumps({"status": "completed", "automatic_qc_passed": True, "run": 1}), encoding="utf-8")
            ukf_qc.write_text(json.dumps({"status": "completed", "automatic_qc_passed": True, "run": 1}), encoding="utf-8")
            wmql_qc.write_text(
                json.dumps({
                    "status": "completed",
                    "version": WMQL_QC_VERSION,
                    "automatic_qc_passed": True,
                    "output_summary": {
                        "bundle_files": [af_live.as_posix()],
                        "bundle_count": 1,
                        "bundle_readable_count": 1,
                        "bundle_nonempty_count": 1,
                        "bundles": [{"query_name": "af.left", "streamline_count": 50}],
                    },
                }),
                encoding="utf-8",
            )

            run1_manifest = {
                "kind": "slicerddsurfer_product_orchestration",
                "version": 1,
                "status": "completed",
                "run_id": "Test1:run1",
                "subject_id": "sub01",
                "subject_root": root.as_posix(),
                "output_root": layout.outputs.as_posix(),
                "run_name": "Test1",
                "tasks": [
                    {
                        "name": "microstructure.dti",
                        "status": "completed",
                        "qc_path": dti_qc.as_posix(),
                        "outputs": {"fa": fa_live.as_posix()},
                    },
                    {
                        "name": "tractography.ukf",
                        "status": "completed",
                        "qc_path": ukf_qc.as_posix(),
                        "outputs": {"tractogram": ukf_live.as_posix()},
                    },
                    {
                        "name": "bundles.wmql",
                        "status": "completed",
                        "qc_path": wmql_qc.as_posix(),
                        "outputs": {"output_dir": layout.query_bundle_extraction.as_posix()},
                    },
                ],
            }
            out_fps, qc_fps = build_manifest_artifact_fingerprints(run1_manifest, run_id="Test1:run1")
            run1_manifest["output_fingerprints"] = out_fps
            run1_manifest["qc_fingerprints"] = qc_fps
            live_manifest_path = layout.run_metadata / "Test1_orchestration_manifest.json"
            live_manifest_path.write_text(json.dumps(run1_manifest, indent=2), encoding="utf-8")

            # Run 2 starts with the SAME run name -> archives Run 1.
            archived_records = _archive_existing_run_records(layout, "Test1")
            archived_manifest_path = archived_records["orchestration_manifest"]
            self.assertTrue(archived_manifest_path.is_file())
            history_dir = archived_manifest_path.parent

            # No binary product files duplicated into history/!
            history_files = list(history_dir.rglob("*"))
            self.assertFalse(any(p.suffix in {".gz", ".vtk", ".trk"} for p in history_files))
            self.assertTrue((history_dir / "snapshots").is_dir())

            # Run 2 now OVERWRITES live FA + live WMQL bundle AND overwrites live DTI & WMQL QC with FAILED QC!
            # (UKF is untouched by Run 2.)
            fa_live.write_bytes(b"FA-RUN2-CORRUPTED")
            af_live.write_bytes(b"WMQL-AF-RUN2-BAD!")
            dti_qc.write_text(json.dumps({"status": "failed", "automatic_qc_passed": False, "run": 2}), encoding="utf-8")
            wmql_qc.write_text(
                json.dumps({
                    "status": "failed",
                    "version": WMQL_QC_VERSION,
                    "automatic_qc_passed": False,
                    "error": "Run 2 failed QC",
                    "output_summary": {"bundle_files": [af_live.as_posix()], "bundle_count": 1},
                }),
                encoding="utf-8",
            )

            # Inspecting Run 1's archived manifest:
            archived_payload = json.loads(archived_manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(archived_payload.get("archived"))

            # 1) Historical QC summary reads Run 1's immutable snapshots -> 3 passed, 0 issues (NOT Run 2's failed QC!).
            qc_summary = recorded_qc_summary(archived_payload)
            self.assertEqual(qc_summary, {"passed": 3, "issues": 0, "unavailable": 0})

            # 2) result_candidates returns ONLY untouched UKF; overwritten FA & WMQL bundle are skipped as 'superseded'!
            candidates, skipped = result_candidates(archived_payload, allow_bundle_directory_scan=False)
            self.assertEqual([c["path"] for c in candidates], [str(ukf_live.resolve())])
            superseded_paths = {
                str(Path(item["path"]).resolve())
                for item in skipped
                if item.get("reason_category") == "superseded" and item.get("path")
            }
            self.assertEqual(superseded_paths, {str(fa_live.resolve()), str(af_live.resolve())})

            # 3) Even with include_qc_not_passed=True (Results tree mode), overwritten outputs MUST stay blocked!
            tree_cands, tree_skipped = result_candidates(
                archived_payload, allow_bundle_directory_scan=False, include_qc_not_passed=True
            )
            self.assertEqual([c["path"] for c in tree_cands], [str(ukf_live.resolve())])
            self.assertEqual(
                {str(Path(s["path"]).resolve()) for s in tree_skipped if s.get("reason_category") == "superseded"},
                {str(fa_live.resolve()), str(af_live.resolve())},
            )

            # 4) automatic_result_candidates also never loads Run 2's overwritten FA or bundle!
            auto_selected, _auto_skipped, _auto_deferred = automatic_result_candidates(archived_payload)
            self.assertEqual([c["path"] for c in auto_selected], [str(ukf_live.resolve())])
            self.assertEqual(manifest_output_status(archived_payload, fa_live), "superseded")
            self.assertEqual(manifest_output_status(archived_payload, ukf_live), "current")

    def test_6_unarchived_different_run_name_overwrite_marks_qc_unavailable_and_blocks_output(self) -> None:
        from shared.result_scene import (
            build_manifest_artifact_fingerprints,
            recorded_qc_summary,
            result_candidates,
        )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve() / "sub01"
            fa = root / "outputs" / "02_microstructure" / "diffusion_tensor" / "sub01_FA.nii.gz"
            qc = root / "cache" / "stages" / "02_microstructure" / "dti" / "dti_qc.json"
            fa.parent.mkdir(parents=True)
            qc.parent.mkdir(parents=True)
            fa.write_bytes(b"RUN-A-FA")
            qc.write_text(json.dumps({"status": "completed", "automatic_qc_passed": True}), encoding="utf-8")

            run_a = {
                "kind": "slicerddsurfer_product_orchestration",
                "version": 1,
                "status": "completed",
                "run_id": "runA:1",
                "subject_root": root.as_posix(),
                "output_root": (root / "outputs").as_posix(),
                "tasks": [
                    {
                        "name": "microstructure.dti",
                        "status": "completed",
                        "qc_path": qc.as_posix(),
                        "outputs": {"fa": fa.as_posix()},
                    }
                ],
            }
            out_fps, qc_fps = build_manifest_artifact_fingerprints(run_a, run_id="runA:1")
            run_a["output_fingerprints"] = out_fps
            run_a["qc_fingerprints"] = qc_fps

            # Run B (different run name) overwrites both live FA and live QC with failure.
            fa.write_bytes(b"RUN-B-FA")
            qc.write_text(json.dumps({"status": "failed", "automatic_qc_passed": False}), encoding="utf-8")

            # Run A's QC summary must report unavailable (NOT Run B's failed QC as issues!).
            self.assertEqual(recorded_qc_summary(run_a), {"passed": 0, "issues": 0, "unavailable": 1})
            cands, skipped = result_candidates(run_a)
            self.assertEqual(cands, [])
            self.assertEqual(len(skipped), 1)
            self.assertEqual(skipped[0]["reason_category"], "superseded")

    def test_6_large_file_sampled_fingerprint_catches_overwrite(self) -> None:
        import shared.result_scene as rs

        with tempfile.TemporaryDirectory() as td:
            big = Path(td) / "tract.vtk"
            data_v1 = b"A" * 200_000
            big.write_bytes(data_v1)
            with mock.patch.object(rs, "_ARTIFACT_FULL_HASH_MAX_BYTES", 1024):
                fp = rs.compute_artifact_fingerprint(big, run_id="r1")
                self.assertIn("sample_sha256", fp)
                self.assertNotIn("sha256", fp)
                self.assertTrue(rs.artifact_fingerprint_matches(fp, big))
                # Overwrite with same size, different content
                big.write_bytes(b"B" * 200_000)
                self.assertFalse(rs.artifact_fingerprint_matches(fp, big))

    def test_6_end_to_end_orchestrator_rerun_same_run_name_isolates_archived_connectome(self) -> None:
        from shared.io_layout import subject_layout
        from shared.orchestrator import ProductOrchestratorConfig, _run_stage_dirs, run_subject_products
        from shared.result_scene import recorded_qc_summary, result_candidates

        with tempfile.TemporaryDirectory() as td:
            root = _write_minimal_raw_subject(Path(td).resolve() / "sub01")
            layout = subject_layout(root)
            parcel = _write_two_edge_parcellation(layout.anatomy_input_dir("ddparcel") / "sub01-DDSurfer-wmparc.nii.gz")
            dirs = _run_stage_dirs(layout, "same_run", root)
            tck = _write_two_streamline_tck(dirs["tractogram_dir"] / "sub01_csd_act_ifod2.tck", swapped=False)

            cfg = ProductOrchestratorConfig(
                subject_id="sub01",
                subject_root=root,
                targets=("connectome",),
                run_name="same_run",
                tractography_backend="csd_act",
                connectome_backend="python",
                connectome_metrics="fiber_count",
                closure_applied=True,
                execute=True,
                resume=False,
            )
            with mock.patch("shared.orchestrator._resolve_parcellation", return_value=parcel):
                res1 = run_subject_products(cfg)
            m1 = json.loads(res1.run_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(m1["status"], "completed")
            self.assertIn("output_fingerprints", m1)
            self.assertIn("qc_fingerprints", m1)
            matrix_live = Path(next(t for t in res1.tasks if t.name == "connectome.streamline_count").outputs["matrix_path"])
            qc_live = Path(next(t for t in res1.tasks if t.name == "connectome.streamline_count").qc_path)
            self.assertTrue(matrix_live.is_file())

            # Run 2 with the SAME run name overwrites the live matrix & fails QC!
            def _run2_failing_connectome(ccfg, *, log=None):
                ccfg.output_dir.mkdir(parents=True, exist_ok=True)
                stem = f"{ccfg.subject_id}-connectome-{ccfg.metric}"
                mat = ccfg.output_dir / f"{stem}.csv"
                nodes = ccfg.output_dir / f"{ccfg.subject_id}-connectome-nodes.tsv"
                qc_p = ccfg.output_dir / f"{stem}_qc.json"
                man_p = ccfg.output_dir / f"{stem}_manifest.json"
                mat.write_text("source_node,target_node,streamline_count\n10,20,999\n", encoding="utf-8")
                nodes.write_text("node_id\tlabel\tname\n1\t10\tX\n", encoding="utf-8")
                qc_p.write_text(json.dumps({"status": "failed", "automatic_qc_passed": False, "error": "run2 bad"}), encoding="utf-8")
                man_p.write_text(json.dumps({"status": "failed"}), encoding="utf-8")
                # Also overwrite published matrix to simulate an in-place clobber before failure
                matrix_live.write_text("source_node,target_node,streamline_count\n10,20,999\n", encoding="utf-8")
                return types.SimpleNamespace(
                    status="failed",
                    qc_path=qc_p,
                    manifest_path=man_p,
                    matrix_path=mat,
                    nodes_path=nodes,
                    node_parcellation_path=None,
                )

            with mock.patch("shared.orchestrator._resolve_parcellation", return_value=parcel), mock.patch(
                "connectome.run_connectome", side_effect=_run2_failing_connectome
            ):
                res2 = run_subject_products(cfg)

            m2 = json.loads(res2.run_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(m2["status"], "failed")
            archived_m1_path = Path(m2["previous_run_records"]["orchestration_manifest"])
            archived_m1 = json.loads(archived_m1_path.read_text(encoding="utf-8"))

            # Live QC is now failed, yet Run 1's archived manifest still shows Run 1's PASSED QC snapshot!
            self.assertEqual(json.loads(qc_live.read_text(encoding="utf-8"))["status"], "failed")
            self.assertEqual(recorded_qc_summary(archived_m1), {"passed": 1, "issues": 0, "unavailable": 0})
            # And Run 1's archived manifest REFUSES to load Run 2's overwritten matrix (marked superseded)!
            cands, skipped = result_candidates(archived_m1)
            self.assertNotIn(str(matrix_live.resolve()), [c["path"] for c in cands])
            self.assertTrue(
                any(
                    s.get("reason_category") == "superseded"
                    and s.get("path")
                    and Path(s["path"]).resolve() == matrix_live.resolve()
                    for s in skipped
                )
            )


class UpdateRollbackVerificationTests(unittest.TestCase):
    """[Item 7] Windows update rollback must never report success or discard journal when locked files fail."""

    @staticmethod
    def _dead_pid() -> int:
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait(timeout=10)
        return proc.pid

    def _make_interrupted_install(self, root: Path) -> tuple[Path, Path, Path]:
        from shared.update_journal import JOURNAL_DIRNAME, JOURNAL_FILENAME

        (root / "shared").mkdir(parents=True, exist_ok=True)
        (root / "SlicerDSpace.py").write_text("VALUE = 1\n", encoding="utf-8")
        helper = root / "shared" / "helper.py"
        helper.write_text("VER = 'new_v2'\n", encoding="utf-8")
        created = root / "shared" / "new_feature.py"
        created.write_text("NEW = True\n", encoding="utf-8")

        jdir = root / JOURNAL_DIRNAME
        (jdir / "backup").mkdir(parents=True, exist_ok=True)
        (jdir / "backup" / "00000.bak").write_text("VER = 'old_v1'\n", encoding="utf-8")
        journal = {
            "kind": "slicerdspace_update_journal",
            "version": 1,
            "pid": self._dead_pid(),
            "started": time.time() - 60.0,
            "target_version": "2.0.0",
            "backed_up": ["shared/helper.py"],
            "created": ["shared/new_feature.py"],
            "backup_map": {"shared/helper.py": "00000.bak"},
        }
        (jdir / JOURNAL_FILENAME).write_text(json.dumps(journal), encoding="utf-8")
        return helper, created, jdir

    def test_7_locked_created_file_fails_rollback_keeps_journal_and_retries_after_unlock(self) -> None:
        from shared.update_journal import JOURNAL_FILENAME, recover_interrupted_update

        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            helper, created, jdir = self._make_interrupted_install(root)

            if sys.platform == "win32":
                # Genuine OS-level Windows file handle lock (Python open() omits FILE_SHARE_DELETE -> WinError 32).
                lock_fh = created.open("rb")
                ctx = mock.MagicMock(__enter__=lambda s: None, __exit__=lambda s, *a: None)
            else:
                lock_fh = None
                orig_unlink = Path.unlink

                def _deny_created(self_path, *args, **kwargs):
                    if self_path.resolve() == created.resolve():
                        raise PermissionError("locked file")
                    return orig_unlink(self_path, *args, **kwargs)

                ctx = mock.patch.object(Path, "unlink", autospec=True, side_effect=_deny_created)

            try:
                with ctx:
                    result = recover_interrupted_update(root)
            finally:
                if lock_fh is not None:
                    lock_fh.close()

            self.assertIsNotNone(result)
            self.assertEqual(result["status"], "rollback_incomplete")
            self.assertFalse(result["ok"])
            self.assertEqual([item["path"] for item in result["failed_removals"]], ["shared/new_feature.py"])
            self.assertTrue(result["errors"])
            # Journal AND backup MUST still exist so recovery can be retried once the lock clears!
            self.assertTrue((jdir / JOURNAL_FILENAME).is_file())
            self.assertTrue((jdir / "backup" / "00000.bak").is_file())
            # Unlocked backed-up file WAS restored cleanly.
            self.assertEqual(helper.read_text(encoding="utf-8"), "VER = 'old_v1'\n")
            # Locked created file is still on disk (we did NOT falsely claim it was removed!).
            self.assertTrue(created.is_file())

            # Retry after lock is released -> completes cleanly and discards journal!
            retry = recover_interrupted_update(root)
            self.assertEqual(retry["status"], "rolled_back")
            self.assertTrue(retry["ok"])
            self.assertFalse(created.exists())
            self.assertEqual(helper.read_text(encoding="utf-8"), "VER = 'old_v1'\n")
            self.assertFalse(jdir.exists())

    def test_7_locked_backed_up_file_fails_restore_keeps_journal_and_retries_after_unlock(self) -> None:
        from shared.update_journal import JOURNAL_FILENAME, recover_interrupted_update

        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            helper, created, jdir = self._make_interrupted_install(root)

            if sys.platform == "win32":
                # Genuine OS-level Windows lock on the backed-up file blocks os.replace() with WinError 5/32!
                lock_fh = helper.open("rb")
                ctx = mock.MagicMock(__enter__=lambda s: None, __exit__=lambda s, *a: None)
            else:
                lock_fh = None
                orig_replace = os.replace

                def _deny_helper(src, dst):
                    if Path(dst).resolve() == helper.resolve():
                        raise PermissionError("locked restore target")
                    return orig_replace(src, dst)

                ctx = mock.patch("shared.update_journal.os.replace", side_effect=_deny_helper)

            try:
                with ctx:
                    result = recover_interrupted_update(root)
            finally:
                if lock_fh is not None:
                    lock_fh.close()

            self.assertIsNotNone(result)
            self.assertEqual(result["status"], "rollback_incomplete")
            self.assertFalse(result["ok"])
            self.assertEqual([item["path"] for item in result["failed_restores"]], ["shared/helper.py"])
            self.assertTrue((jdir / JOURNAL_FILENAME).is_file())
            self.assertTrue((jdir / "backup" / "00000.bak").is_file())
            # Unlocked newly created file was removed, while locked backed-up file stayed v2.
            self.assertFalse(created.exists())
            self.assertEqual(helper.read_text(encoding="utf-8"), "VER = 'new_v2'\n")

            # Retry after lock released -> restores helper.py to v1 and cleans up journal!
            retry = recover_interrupted_update(root)
            self.assertEqual(retry["status"], "rolled_back")
            self.assertTrue(retry["ok"])
            self.assertEqual(helper.read_text(encoding="utf-8"), "VER = 'old_v1'\n")
            self.assertFalse(jdir.exists())

    def test_7_code_updater_preserves_journal_when_mid_apply_rollback_fails(self) -> None:
        from shared.code_updater import JOURNAL_DIRNAME, download_and_apply_code_overlay
        from shared.update_journal import JOURNAL_FILENAME, recover_interrupted_update
        from Testing.Python.test_b4_w4_platform import _updater_fixture

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td).resolve()
            inst, orig_sha, _v2, manifest, offline = _updater_fixture(td_path)
            lock_handles = []

            def _hook_fail_after_writing_helper(rel: str) -> None:
                if rel == "SlicerDSpace.py":
                    # Lock shared/helper.py (which was already replaced with v2!) then fail apply!
                    if sys.platform == "win32":
                        lock_handles.append((inst / "shared" / "helper.py").open("rb"))
                    raise OSError("simulated failure after helper updated")

            try:
                if sys.platform == "win32":
                    with self.assertRaisesRegex(RuntimeError, "rollback was incomplete"):
                        download_and_apply_code_overlay(
                            module_root=inst,
                            manifest=manifest,
                            offline_dir=offline,
                            _apply_hook=_hook_fail_after_writing_helper,
                        )
                else:
                    orig_replace = os.replace

                    def _deny_rbk(src, dst):
                        if ".rbk." in str(src):
                            raise PermissionError("locked")
                        return orig_replace(src, dst)

                    with mock.patch("shared.update_journal.os.replace", side_effect=_deny_rbk):
                        with self.assertRaisesRegex(RuntimeError, "rollback was incomplete"):
                            download_and_apply_code_overlay(
                                module_root=inst,
                                manifest=manifest,
                                offline_dir=offline,
                                _apply_hook=_hook_fail_after_writing_helper,
                            )
                # Journal and backup MUST be preserved (NOT discarded)!
                jdir = inst / JOURNAL_DIRNAME
                self.assertTrue((jdir / JOURNAL_FILENAME).is_file())
                self.assertTrue((jdir / "backup").is_dir())
            finally:
                for fh in lock_handles:
                    fh.close()

            # Once lock is released, startup recovery finishes rollback back to original tree!
            from shared.code_updater import compute_code_tree_sha256

            res = recover_interrupted_update(inst)
            self.assertEqual(res["status"], "rolled_back")
            self.assertTrue(res["ok"])
            self.assertEqual(compute_code_tree_sha256(inst)[0], orig_sha)
            self.assertFalse((inst / JOURNAL_DIRNAME).exists())


if __name__ == "__main__":
    unittest.main()
