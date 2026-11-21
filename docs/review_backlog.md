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
| Opus P2 (installer prune) | P2 | `Testing/Python/test_installer_prune_fallback.py:257-268` checks the release-package stub against tag `v0.1.1`, but the published base packages are built from fresh HEAD code (`repack-os-base`; last os-bases sync 5070ca29, parent 2e8f7315). CI checks out no tags, so this check is skipped there (+1 skipped per job in run 44887655250). | Name the repack source commit in the test, or check it against a ref that CI fetches. |
| Opus P2 (installer prune) | P2 | `Testing/Python/test_installer_prune_fallback.py:202-211`: when `node` is not on PATH, the two JS copies are dropped without a message and the test reports OK on the Python copy only (same pattern as `_InstallerCopies`). The matrix workflow has no setup-node step. | Skip with a message, or fail when `CI` is set and node is missing. |
