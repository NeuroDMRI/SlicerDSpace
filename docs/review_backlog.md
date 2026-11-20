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
