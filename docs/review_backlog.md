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
