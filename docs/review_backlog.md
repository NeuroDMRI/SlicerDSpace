# Review backlog (non-blocking items)

Items that independent reviews found but that do not make a result wrong. They do not start a new review round.

## 4 November 2026: new promo film on the site (Opus + Codex)

Scope: the 64.8 s film, its site copy, receipts, checks and docs. Codex: one P2 (FACTS §9 black-segment claim),
fixed in the same commit. Opus: no blocking item; two items in the new text (SPEC §5.3 label, Addendum 7 wording)
were fixed in the same commit; the items below were already present before this change.

| Source | Label | Item | Proposed fix |
|---|---|---|---|
| Opus s1104 P2-1 | P2 | `check_site.py --film` compares the site copy's SHA-256 with the given file, but the site copy has been a CRF 24 encode since 30 October 2026, so `--film <master>` always fails and `--film <site copy>` proves only the byte count. SPEC §7.1 item 4 describes the old check. The hash is still bound by `test_website_media` (receipt) and `release_audit.py`. | Compare the site copy's SHA-256 with `published_assets.json`, or drop `--film` and update SPEC §7.1 item 4. |
| Opus s1104 P3-1 | P3 | SPEC §7.1 item 7 says "13 cue texts" and names `transcript_matches_srt`; the check is now `minimal_film_controls` and asserts that no transcript exists. | Rewrite item 7 to describe `minimal_film_controls`. |
| Opus s1104 P3-2 | P3 | SPEC §7.2 item 4 says `avc1.640032` (High, level 5.0); ffprobe reports level 4.1 (`avc1.640029`) for the site copy. `canPlayType` passes either way. | Change the text to `avc1.640029` (High, level 4.1). |
| Opus s1104 P3-4 | P3 | The maintenance note in `docs/website/README.md` (and `README.zh-CN.md`) lists only the bytes and hash pins. A new film also needs the duration in `check_site.py` (`film_probe`), the duration in `browser_check.mjs`, and the frame count in `Testing/Python/test_website_media.py`. | List all film pins in both READMEs. |

## 21 November 2026: MPS / Rosetta / compatibility fixes, batch 1 (Opus)

Scope: branch `fix/mps-rosetta-compat` (RAM helper, surface device report, arm64 exec, UKF VTK check, Slicer 5.10+,
real-MPS tests). Opus: one blocking item (stale `UPSTREAM.json` hashes), fixed. P2-4 (strict VTK set match) and P2-6
(UKF error remedy) were fixed in the same fix round with tests. Decide by 5 December 2026.

| Source | Label | Item | Proposed fix |
|---|---|---|---|
| Opus s1121 P2-2 | P2 | MRtrix Python scripts (`dwi2response`, `5ttgen` in `tractography/csd.py:154-170`, `dwifslpreproc`) start their MRtrix binaries from a translated process, so those still run x86_64; `shared/input_import.py:162` (`mrconvert`) is not wrapped. Speed only. | Run the scripts with an arm64 interpreter, or set an arch preference for their children; wrap `input_import` mrconvert. |
| Opus s1121 P2-3 | P2 | No automatic fallback when an arm64 slice cannot load; recovery needs `SLICERDSPACE_DISABLE_ARM64_EXEC=1`. | Name the kill switch in the `CalledProcessError` message, or retry once without the prefix. |
| Opus s1121 P2-5 | P2 | `_running_vtk_version()` reads the current interpreter's `vtk`; a headless non-Slicer Python with another pip `vtk` can reject a working cli-modules UKF. | Read the VTK version from Slicer's lib folder, or skip the check outside Slicer. |
| Opus s1121 P2-7 | P2 | `anatomy/surface/pipeline.py` is in the shared-raw surface `input_sha256`, so the first surface run after the overlay recomputes DDSurfer. | Say so in the release note. |

## 21 November 2026: batches 2 and 4 (preview, FSL gate, liveness, early errors, dependencies)

Scope: batch 2 (`a745d24a`) and batch 4 on branch `fix/mps-rosetta-compat`. Every blocking item and every P2 from the
Opus and Codex rounds was fixed in the fix rounds, with tests. These items stay open. Decide by 5 December 2026.

| Source | Label | Item | Proposed fix |
|---|---|---|---|
| W6 D8 | Fixed | **Fixed**: `scripts/package_slicer_extension.py` now writes `"arch"` and `"slicer_numpy_minimum": "2.0.2"` to `python-runtime.json`. | Keep in sync with future Slicer/torch release floors. |
| Audit LE-7 | P2 | `msmt_csd` on single-shell data is a method choice, not a compatibility item. | Decision for the spec owner. |
| W6 LE-3 | P2 | The surface resume key compares `device` and `resolved_device`; a run that fell back to CPU may not be reused by a later `auto` run. Not measured. | Measure whether it forces a recompute; drop `resolved_device` from the key only if the outputs are equivalent. |
| W1 H4 | P2 | On macOS the CTK launcher turns a signal kill into exit code 1, so the kill message has no signal number. Linux and Windows launcher exit codes are not checked on real systems. | Check in the GitHub Actions matrix. |
| MPS probe | P2 | MPS batch-size scaling (DDSeg, DDParcel, PNL mask, TractCloud) gives at most 1.16x; vf_net Conv3d emulation needs 6.45 GiB, over the 4.87 GiB budget; Gibbs matmul DFT gives 1.1x. | No change. The real gain needs a native arm64 worker with a newer torch, or a native arm64 Slicer. |
| Opus r3 P2-1 | P2 | On a mount whose `flock` fails with an errno outside EAGAIN/EACCES/EDEADLK (ENOLCK/ENOTSUP on some NFS/SMB), every overlay apply refuses with "check that the extension folder is writable". | Put the errno in the message, or say that file locking is not supported there. |
| Opus r3 P2-2 | P2 | `--jobs 1` does not clear an inherited `SLICERDSPACE_PARALLEL_SUBJECTS`; in a `--jobs > 1` run the last subject can run alone and still skips the LE-3 GPU check. | Pop the variable when one subject runs; accept the tail case. |

## 21 November 2026: WMA resume must not accept a killed rerun

Scope: branch `fix/wma-stale-resume`. Round 1: Opus no blocking item; Codex 1 item (Windows skip for a new test), fixed.
These items stay open. Decide by 5 December 2026.

| Source | Label | Item | Proposed fix |
|---|---|---|---|
| Opus P2 | P2 | `scripts/audit_bundle_products.py --update-qc` is a second writer of the WMA manifest status. On a killed-rerun folder it can set `status='completed'` when the new input contains every old streamline; `_wma_complete` then accepts the old products. Plain resume cannot reach this state; it needs this manual step. | In `update_qc_records`, refuse a manifest whose status is `running`. |
| Codex / author | P2 | `test_resume_does_not_reuse_legacy_output_for_native_org` (older test, same POSIX fixture) has no Windows skip; with `tractography.wma.sys.platform` patched to `win32` it gives errors=1 ("Windows WMA requires the managed WSL command bridge"). | Add the same `skipIf(win32)` decorator. |
| Author | P2 | `run_wma_org_atlas` leaves the previous run's QC (`status: completed`) on disk while a rerun is in progress. The resume check no longer trusts it, but another reader of the QC file could. | Write a `running` QC at the start of an executed run, after `archive_run_records`. |

## 21 November 2026: CI matrix, installer reinstall (I1), Linux glibc floor (G1)

Scope: branch `ci/compat-matrix-fix`. Every blocking item from the Opus and Codex rounds was fixed, with tests. These items
stay open. Decide by 5 December 2026.

| Source | Label | Item | Proposed fix |
|---|---|---|---|
| Opus r2 P2-a | P2 | If helper pass 1 writes `scheduled_uninstall` but Slicer exits non-zero at teardown, the installer stops with "rejected archive". The schedule stays, so the next Slicer start removes the old copy, and the message does not say so. | Go on to pass 2 whenever the receipt says `scheduled_uninstall`, or name the scheduled removal in the error. |
| Opus r2 P2-b | P2 | If pass 2 fails after the start-up removal, the error does not say that the old SlicerDSpace was removed. | Add "the previous SlicerDSpace was removed; run the installer again" to pass-2 errors. |
| Opus r2 P2-c | P2 | A GUI Setup retry in the same session asks for a restart even though this session installed the copy, so the copy is not loaded. | Record in `state` that this session installed the copy, and skip `_handle_installed_copy` in that case. |
| CI macOS | P2 | The report-only unit suite is skipped on macOS full-install: the hosted runner was lost twice in that step (cause not measured). The smoke stays the gate. | Measure runner memory in that step, then turn the suite back on. |
| Opus r3 P2 | P2 | When Setup cannot cancel its own removal, the message says "open Setup again", but the next start removes Setup. | Say: restart 3D Slicer (this removes Setup), then install SlicerDSpaceSetup again and reinstall SlicerDSpace. |
| Codex r3 P2 / CI | Fixed | **Fixed** (run 44860990718, 12/12 jobs): `test_installer_reinstall_i1`, `test_linux_glibc_floor_g1` and `test_installer_gui_reinstall_gap` have 0 failures on Windows, Linux and macOS. The I1 fake expects `os.path.abspath` of the Slicer path and accepts the `xvfb-run -a` prefix; the `os.confstr` patches use `create=True`; the node output is read as UTF-8. | Done. |
| CI unit suites | P2 | Older modules still fail in the report-only unit suite (run 44866252278): Windows repo-only failures=51–52, errors=312; Linux repo-only failures=5, errors=309; macOS repo-only failures=5–6, errors=305. **UnicodeDecodeError part fixed:** the 9 node calls in `test_network_and_on_demand.py` and `test_mps_and_cross_platform.py` read the output as UTF-8; the 13 decode errors per Slicer 5.10.0 Linux/macOS repo-only job (run 44860990718) are now 0 in every job. Known remaining causes include modules that the repo-only environment does not have (for example `psutil`, `nibabel`). | Sort the remaining failures into platform-only or environment skips and real bugs. |
| Opus P2 (test fakes) | Fixed | **Fixed** (run 44864495879, 12/12 jobs): `test_b4_w4_platform._js_payload` reads the node output as UTF-8. With a non-ASCII character in `buildScript`, the ASCII locale gave 8 `UnicodeDecodeError`s before the fix and 0 after. | Done. |
| Opus P2 (test fakes) | Fixed | **Fixed** (run 44864495879, 12/12 jobs): a new I1 test asserts no `xvfb-run` prefix on Linux with `DISPLAY` set, on Windows and on macOS, for both installer copies. A product change that adds the prefix in any of these cases fails the test. | Done. |
| Opus P2 (node encoding) | Fixed | **Fixed**: `build_setup_zips` reads the node output with `encoding="utf-8", errors="replace"`, so a non-ASCII node error reaches the `RuntimeError`. The new test `test_repackage_node_encoding` gives `UnicodeDecodeError` on the old line and passes on the new one. Run 44885046381: 0 failures of this test in all 10 unit-suite logs; the full-install jobs failed on an installer `ImportError` from 801b4ba7, which this change does not touch. | Done. |
| Opus P2 (installer prune) | Fixed | **Fixed** (run 44893183147, 12/12 jobs): the check now reads `shared/user_config.py` at `RELEASE_PACKAGE_SOURCE_COMMIT` (2e8f7315); both matrix jobs fetch that commit after checkout, and the check fails on CI when it is missing. Skipped count is 1 lower in every job; 0 failures. Was: `Testing/Python/test_installer_prune_fallback.py:257-268` checks the release-package stub against tag `v0.1.1`, but the published base packages are built from fresh HEAD code (`repack-os-base`; last os-bases sync 5070ca29, parent 2e8f7315). CI checks out no tags, so this check is skipped there (+1 skipped per job in run 44887655250). | Name the repack source commit in the test, or check it against a ref that CI fetches. |
| Opus P2 (installer prune) | Fixed | **Fixed** (run 44893183147, 12/12 jobs): when node is missing, the JS copies fail on CI ("node is required on CI for the JS installer copies") and are skipped subtests with a message locally; the Python copy still runs. 0 matches of that message in all 12 logs. Was: `Testing/Python/test_installer_prune_fallback.py:202-211`: when `node` is not on PATH, the two JS copies are dropped without a message and the test reports OK on the Python copy only (same pattern as `_InstallerCopies`). The matrix workflow has no setup-node step. | Skip with a message, or fail when `CI` is set and node is missing. |
| Codex P2 (CI truthiness) | P2 | `Testing/Python/test_installer_prune_fallback.py:47-48`: `_on_ci()` treats `CI` values "0", "false", "no" and whitespace as not-CI, so a missing commit or node then skips instead of failing. GitHub Actions sets `CI=true`, so the matrix is not affected (Codex said P1; reclassified: no result changes on the real CI). | Use `bool(os.environ.get("CI"))`, or keep the stricter rule and document it. |

## 21 November 2026: close guard (ask before Slicer closes while a DSpace job runs)

Scope: branch `feat/close-guard`. Round 1: Opus no blocking item; Codex 2 blocking items in the smoke script, both fixed.
Re-review of the fix (Opus and Codex): no blocking item. These items stay open. Decide by 5 December 2026.

| Source | Label | Item | Proposed fix |
|---|---|---|---|
| Opus r2 P2-1 | P2 | `Testing/Python/slicer_close_guard_smoke.py` step (3) (`closeAllWindows`) has no early stop, while steps (1) and (2) have one. After a guard regression on that path the script still reports `passed=false`, but a posted Quit might change the process exit code (not confirmed). | Stop when `close_all_windows_refused` or `helper_running_after_close_all_windows` is false; read the JSON `passed` field as the result of record. |
| Opus r2 P2-2 | P2 | The "Safety order" docstring of the smoke script names 3 preconditions; the code checks 5 (also `guard_installed` and `window_visible_before_close`). | Name all 5 in the docstring. |
| Opus r2 P2-3 | P2 | `close_all_windows_refused` accepts an ask count `>= 1`; the other paths require exactly 1. | Use `== 1` so a repeated question on this path fails the check. |
| Author | P2 | Measured only on macOS, Slicer 5.12.4, Qt 5.15.18, through `window.close()`, the File > Exit action and `closeAllWindows()`. Not measured: a human Cmd+Q, a Dock quit, a logout, and Windows and Linux. | Run `slicer_close_guard_smoke.py` in the GitHub Actions real-Slicer matrix; try Cmd+Q and Dock quit once by hand. |
| Author | P2 | A running update check (a few seconds) counts as a "code update" job, so a close in that window asks the question. | Count the update job only while it downloads or applies, if users report it. |

## 21 November 2026: UI batch, packages A and B (run lock ticks; grey out algorithms the data cannot run)

Scope: branch `feat/ui-batch`. Round 1: package A no item; package B 3 blocking items (F1 gate read raw inputs while the planner reads the preprocessed files; F2 stale auto-import files; F3 uncached folder listing on every event) and 2 P2 items (F4, F5), all fixed in the one fix round.
Re-review of the fix (Opus and Codex): no blocking item. These items stay open. Decide by 5 December 2026.

| Source | Label | Item | Proposed fix |
|---|---|---|---|
| Opus r1 P2 | P2 | `shared/algorithm_availability.py`: unreadable bvals are not cached as a negative result, so each refresh reads the file again. | Cache the "unknown" answer on the same `(path, size, mtime_ns)` key. |
| Codex r1 P2 | P2 | The environment check before a run still asks for the dependencies of an algorithm that the data filter skips. | Run the environment check on the filtered request. |
| Opus r1 P2 | P2 | The iFW echo-time validation still runs when iFW is skipped for data reasons. | Validate the echo times only when iFW stays in the filtered request. |
| Author | P2 | A folder that keeps its convertible source next to an imported `input/` never greys an algorithm, also when the import receipt is current (the gate answers "unknown"). | Expose the importer's "receipt is current" test and use it in the gate. |
| Author | P2 | The `dki_design_rank` code and its Chinese text have no test (no cheap fixture passes the per-shell checks and fails the rank check). | Build a fixture with collinear directions per shell. |
| Author | P2 | Real-Slicer check of package A and B covers only the checkbox style and the "?" marker (light and dark, Slicer 5.12.4). The popup, the summary text and the Run filter are tested with fake Qt only. | Add a picker smoke step to `slicer_algorithm_picker_smoke.py`. |

## 21 November 2026: UI batch, packages C, D7 and D3 (remembered inputs, Refresh, Results tree, Subject ID suffix)

Scope: branch `feat/ui-batch`. Package C (remember the input tiles, Refresh button): round 1 had 3 blocking items (FC1 to FC3), fixed in the one fix round; re-review (Opus and Codex) no blocking item. Package D7 (Results tree, collapse, pop-out window): round 1 had blocking items FD1 to FD7; the fix of FD3 introduced FD8 (a tree item used after the context menu closed), fixed in the allowed third round; re-review no blocking item. Package D3 (Subject ID suffix when the input changed): round 1 had 3 blocking items (FE1 environment-check files made a fresh folder get a suffix; FE2 the detached data file was part of the input identity; FE3 a reused DICOM series skipped validation) and FE4 (suffix notice replaced by the next status), all fixed in the one fix round.
These items stay open. Decide by 5 December 2026.

| Source | Label | Item | Proposed fix |
|---|---|---|---|
| Opus r1 P2 (C) | P2 | A scene loaded from a file leaves the input tiles empty, and the next save overwrites `recent.input_sources` with empty values. | Do not save empty tiles when the restore was skipped for a loaded scene. |
| Author (C) | P2 | The "Selected products" summary, the Run-button tooltip and the setup dialog still list also-run algorithms from the unfiltered closure. | Use the data-filtered closure in these three places, as the also-runs note does. |
| Opus r1 P2 (D7) | P2 | `_load_background_manifest_result` leaves the run state "running" when the manifest JSON cannot be read. | Set the state to "failed" in the read-error branch. |
| Opus r1 P2 (D7) | P2 | `Testing/Python/slicer_gui_operations_smoke.py` depends on the remembered result folders of the machine. | Point the smoke test at a scratch settings file. |
| Opus r1 P2 (D7) | P2 | A refresh builds the selected subject twice. | Build once and reuse the items for the selection. |
| Author (D7) | P2 | A job that ends through the "manifest" outcome with a manifest that still says "running" becomes "completed", and the subject keeps "running or interrupted". | Read the final manifest status before the state is set. |
| Author (D7) | P2 | Real-Slicer coverage: the tree colours were not checked in a Slicer that starts in dark mode; the context menu and the run-state transitions are tested with fake Qt only. | Add a dark-start smoke step and a context-menu step. |
| Opus r1 P2 (D3) | P2 | A detached-NRRD import made before this change saved the signature of a temporary header. The first re-import of the same header after the update goes once to `<ID>_2`; nothing is overwritten. | Apply the legacy rule when the manifest has `plan.detached_nrrd` and its saved signature does not name the real header. |
| Opus r1 P2 (D3), measured | P2 | When the Output Folder is the subject folder itself (`<X>/<ID>`), the suffixed subject is created inside it (`<X>/<ID>/<ID>_2`). Measured in Slicer 5.12.4. Nothing is overwritten. | Create the suffixed folder beside the base folder and move the Output Folder with it. |
| Author (D3) | P2 | A plain NHDR import signs only the header, not the data file it names. An edited data file under the same header counts as the same input (gap older than D3). A changed "Detached data" file is converted again but stays in the same subject. | Sign the named data file in every NHDR import path and accept old manifests once. |
| Author (D3) | P2 | The environment-check files are ignored only under `outputs/00_run_metadata/environment`. If the Output Folder text ends in `<ID>/output` (singular), the check writes under `output/00_run_metadata/environment`, and a first import after the check still gets a suffix. | Take the ignored folder from the same function the GUI uses for the report folder. |
| Author (D3) | P2 | The automatic DICOM import now also gets a DICOM signature and an intent marker (one more stat pass; the marker is listed in the receipt). | Limit the marker and the signature to tile imports. |
| Author (D3) | P2 | Real-Slicer check of D3 covers the NIfTI tile import (button click), the restart restore and the nested Output Folder. The DICOM and NRRD paths, the Run click after a suffixed import and the suffix notice are tested with fake Qt only. | Add an import step with a DICOM series and a Run step to a smoke script. |
