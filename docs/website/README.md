<div align="center">

# Project Website

**Static website assets and maintenance guide**

[Project](../../README.md) | [Website](https://neurodmri.github.io/SlicerDSpace/) | [Original Notes](UPSTREAM_README.md) | [中文](README.zh-CN.md)

</div>

---

## Source And Attribution

The website was contributed by [xiaji](https://github.com/xiaji), whose original
Git author is `x1aji <jwjwei@gmail.com>`. Its three original commits
`935a5d9`, `444d8db` and `5b09bc2` are retained as ancestors of `main` through a
normal merge, not a squash, rebase or copied replacement commit. Their author,
committer, timestamps and co-author trailers are unchanged.

Public HTML, styles, scripts, fonts, images, subtitles and the 47,838,648-byte
promo film live under `docs/site/`. Integration moves the original files without
redesigning the site or re-encoding its media. [FACTS.md](FACTS.md),
[SPEC.md](SPEC.md) and [UPSTREAM_README.md](UPSTREAM_README.md) preserve the
contributor's historical source, design and review records. They are not current
installation instructions and are not part of the public deployment.

---

## Performance Section

The performance comparison table in `#performance` follows the `#runtime`
stage-time figure and shows all 14 timing rows by default. The top navigation
and footer link lead directly to the table, without an outer disclosure.
Only the method and verification details are initially collapsed.
Clicking the plus button or its algorithm name expands those details directly
underneath that measurement row; clicking again collapses it. There is no
separate methods list below the table. Native button keyboard activation is
supported. Detail rows carry the HTML `hidden` attribute and stay collapsed even
if JavaScript or styles fail to load. Buttons are enabled after initialization;
the optimization report and JSON download remain available without scripting.
Reloading the page resets all details to collapsed, including when the URL
retains an old record fragment.
`node Testing/Website/browser_check.mjs --performance` checks this table's
initial state, plus/minus controls, keyboard use and script/style failure cases.
There is no separate runtime or performance page; the legacy redirect was removed.
The evidence separates equivalent
implementation changes, native threads, workflow scheduling and Python-port controls.
Each record states its reference, measurement scope, output-equivalence checks and
limitations; no whole-workflow benefit is claimed.

The October 30 refresh adds the October 29 MK-Curve signal-reuse experiment.
Its row compares the preceding qualified-QR Python path with the incremental
change. Expanded details and the JSON retain both solver outcomes; the batched
candidate has no consistent benefit and is not the default. Catalogue review
dates are generated independently of the dates on individual measurements.

`python3 scripts/build_performance_site.py` exports selected benchmark records and
renders their tables into `index.html`, between the `measured-evidence` markers.
`--check` verifies source hashes and detects stale HTML/JSON. Only the curated
`assets/data/performance.json` is public. Raw runtime snapshots, private filesystem
paths, subject data and internal reports are not copied into the site. The new files
and generator are excluded from the extension runtime.

The page has no performance filters, measurement counter, highlighted bar cards
or optimization-table image. The complete table retains each record's method,
measurement scope, timings and type of change. The editable optimization figure
and scientific benchmark records remain in the repository for other uses.

---

## Agent-Assisted Setup

The homepage's Get started section (`#start`) includes the project's
install-and-use Agent Skill, its usage reference, CLI recipes and two copyable
prompts: installation and analysis, without the earlier capability panels.
The Agent route uses the same left-hand numbered rail as manual setup: read the
Skill, install/enable, and run an existing installation. Its counter resets
independently from the manual route; both copyable prompts remain unchanged.
The CLI and cohort instructions are always expanded. Existing `#skills` links still reach these
prompts within Get started. The prompts are also in both root READMEs
and the manuscript's setup instructions. The Skill contains
Agent-facing operating instructions rather than user prompts. Model downloads follow the user's confirmed storage choice; an
analysis request waits for plan confirmation. Skill files stay in the repository.
The Skill runs through the user's Agent. The website bot offers local, curated
bilingual answers and links to these prompts. It stays floating on every screen
size, with drag support, gaze motion and a viewport-fitted dialog; it does not
move into the header on small screens. It makes no network requests and does not
persist questions. `node Testing/Website/guide_bot_check.mjs` checks interactions,
dragging, viewport bounds, keyboard access and the no-JavaScript fallback.

---

## Figures, Methods And References

Scientific figures and the video have no added rounded frames. Result images use
open, transparent plates with shallow curved separators and plain method labels.
The gallery has three desktop columns, two tablet columns and one mobile column;
the scientific image files and their crop/contain rules are unchanged.

The integration catalogue is the final main-page section. The product map
remains; the six example combinations and standalone In Slicer section are
removed. The hero and footer use the transparent, outlined
`assets/img/slicerdspace-wordmark-light.svg`; the header contains only centered
navigation, with no logo or status badge. Verify the wordmark with
`python3 scripts/check_website_assets.py --check`. Browser icons remain
PNG exports of `paper-latex:docs/figures/logo-ref.svg`; verify them with
`python3 scripts/check_website_assets.py --check`.

- The WMQL gallery tile uses `assets/img/res-wmql-ukf.jpg`, an unchanged copy of
  the official multi-query UKF render in `paper-latex:docs/figures/src/results/`. Its source
  and hash are recorded in `wmql_sources.json`. `python3 scripts/check_website_assets.py --check`
  verifies the published copy; regenerate and import artwork from the independent repository when updating it. The full lateral view fits
  inside the gallery tile without cropping. The earlier CSD/ACT AF/CST thumbnail is retired.
- `assets/img/fig-product-map.png` (section `#choose`) and
  `assets/img/fig-runtime-resources.png` (section `#runtime`) are rasterised from the
  editable SVG sources in `paper-latex:docs/figures/`, which are not published. See
  [the independent artwork repository](https://github.com/ChengjinLii/paper-latex/blob/main/docs/figures/README.md) to rebuild them. Each `<img>` carries
  the pixel size of its file and links to the full-size file.
- The standalone "What is integrated" section (`#integrated`) lists 35 concrete
  method/utility entries, including the available denoisers, masking alternatives,
  anatomical and microstructural models, tracking, fiber bundles and regional analysis.
  Each entry identifies its implementation, runtime
  requirements. Historical run-specific orange status badges and the optional
  reference-runtime row are removed. Runtime/access requirements remain explicit,
  including the UKF reference route for WMA and Python API scalar/length summaries.
  Detailed provenance and validation records remain linked; the table does not
  certify every output or promote historical failed runs.
- The method table links directly to primary papers by DOI and to official
  software/documentation URLs, with full citations available in link titles.
  There is no separate bibliography or numbered reference-link list.
  `docs/website/references.json` retains the reviewed source catalogue;
  publisher-deposited metadata for the added DOI entries are retained in
  `reference_verification_20261026.json`.
  `python3 scripts/build_website_catalog.py` renders the method inventory and source links;
  `--check` catches stale HTML, missing source entry points and missing citations.
  Bibliography URLs are allowlisted from this reviewed catalogue; other external
  links still require an explicit `ALLOWED_URLS` entry in `Testing/Website/check_site.py`.

The footer echoes the header's 2 px RGB gradient line. A larger linked wordmark
balances two navigation columns, with a separate, quieter copyright/legal row
and back-to-top link. The layout reflows for tablet and phone widths without
boxed cards. `node Testing/Website/browser_check.mjs --footer` checks both
languages, responsive layouts, the gradient and navigation.
Mobile bottom spacing leaves room for the floating guide without hiding links.
Film production and dataset records remain in the
maintainer's source notes instead of a visitor-facing About the film section.
Visitor copy uses the release presentation without repository-access warnings.

The centered hero shows the SlicerDSpace wordmark and two actions, Watch the film
and Get started, followed directly by the film. The introductory subtitle, lead
paragraph and hero status notes are removed. The Idea, Workflows and Status
sections are removed. Subject
identifiers stay in source records instead of repeated gallery introductions.
Get started includes
staging, module paths, extension requirements, canonical inputs, GUI environment
preparation and the first DTI run, with optional CLI/cohort instructions. The
homepage retains the contributor's established theme and original media.

---

## Publishing

The publication source is the root of the dedicated
[`gh-pages` branch](https://github.com/NeuroDMRI/SlicerDSpace/tree/gh-pages).
It contains a locally validated snapshot of **only `docs/site/`**, including
`index.html`, `zh.html` and `.nojekyll`. The software source, internal documents,
models and independent `paper-latex/` repository are not publication inputs.
Website files are also excluded from the Slicer extension's runtime package.

The publication branch starts with a new root commit; it does not restore old
software or manuscript history. Do not merge this generated branch into `main`.

The film stays in ordinary Git for static playback. The repository size audit
permits only this exact public path, byte count and SHA256, not arbitrary large
videos or model files. Any replacement needs a separately reviewed update.

The repository owner must first open
[Settings > Pages](https://github.com/NeuroDMRI/SlicerDSpace/settings/pages)
and choose **Source: Deploy from a branch**, **Branch: gh-pages**, **Folder:
/(root)**, then save. This one-time repository setting is separate from pushing
the branch. The current repository connection supports content operations but
does not expose the Pages administration endpoint.

For subsequent updates, run the local checks below and copy the reviewed site
into an isolated checkout of `gh-pages`, preserving its `.git` directory and
removing obsolete publication files. Check that its tree contains website files
only, commit the snapshot and push that branch. Keep development edits on `main`
separate from this publication step.

The custom `Website` workflow remains manual-only and is not required or
dispatched for this branch-based setup. GitHub manages its own Pages build and
deployment, which can still appear in the Actions interface. Accept publication
only after the public URL responds successfully and matches the snapshot.

Expected URL: `https://neurodmri.github.io/SlicerDSpace/`. The canonical URL,
share tags and 404 home link assume this address. Review the original notes
before changing it. Do not configure `main /docs`: that would expose documents
outside the intended public website.

---

## Local Checks

The header's Eng / 中文 capsule follows the personal homepage's pill styling.
`zh.html` is a static Chinese version, using the same layout, scientific figures,
film and bibliography as `index.html`. Both language links work without JavaScript;
when JavaScript is enabled they preserve the current section. The language control
sits independently at the right, leaving the section navigation centered.

Chinese copy is reviewed in `zh-CN.json`. After changing the English homepage,
run `python scripts/build_website_locales.py` to rebuild the Chinese page.
New untranslated text blocks generation; `--check` detects stale localized HTML.
Shell commands, algorithm identifiers and original reference titles are retained;
Agent prompts, interface controls and the project guide are localized.

```bash
python scripts/build_website_locales.py --check
python scripts/audit_pages_site.py
python -I Testing/Website/check_site.py
npm ci --prefix Testing/Website
node Testing/Website/browser_check.mjs
```

The browser test uses installed Google Chrome by default. `CHROME_EXECUTABLE`
can point to a compatible Chrome for Testing binary. Optional `PLAYWRIGHT_CORE`
selects an existing module; otherwise the pinned local dependency is used.
Video codec/playback checks are still required. Optional ffprobe checks clearly
report SKIP when ffprobe is unavailable.

Screenshots and browser results stay under ignored `Testing/Website/out/`.
Browser dependencies are test-only and do not ship in the website or Slicer
extension. The public-file allowlist rejects private documents and symlinks.

---

## Media Delivery

The film uses `preload="none"` with a poster and starts downloading only after
playback is requested. Keep the poster eager: the video sits near the first fold.
Scientific figures use native `loading="lazy"`, asynchronous decoding, explicit
dimensions and low fetch priority. The Results overview remains the full SVG.

Product-map and runtime PNGs have lossless WebP alternatives in `<picture>`;
the original PNG remains the fallback and full-size download. Regenerate these
with `python scripts/optimize_website_images.py`; `--check` verifies their bytes,
decoded RGBA pixels, dimensions and asset receipts. Their transfer sizes fall
from 186,955 to 73,288 bytes and from 267,350 to 81,220 bytes, respectively.

The film retains 1920 x 1080, 30 fps, all 1,944 frames, its original 64.8-second
duration and the unchanged AAC payload. H.264 CRF 24 with `+faststart` reduces it
from 45,452,363 to 23,356,794 bytes. The recorded full-frame SSIM is 0.993841.
Retain the master outside the public tree; do not repeatedly recompress the
website copy. The maintainer-only builder/verifier is:

```bash
python scripts/optimize_website_video.py --source /path/to/original-master.mp4 \
  --ffmpeg /path/to/ffmpeg
```

`--verify-existing` audits an already encoded copy without re-encoding it.
The asset receipt contains source/output hashes, metadata and verification.
When changing the film, also update the exact reviewed bytes/hash in
`scripts/release_audit.py` and `Testing/Website/check_site.py`.

Browser checks verify zero MP4 requests before playback, working play/replay,
mobile layout, format selection and no-JavaScript controls. The methods table
shows linked reference numbers; full citations remain in link titles.

References: [web.dev video loading](https://web.dev/articles/lazy-loading-video),
[MDN image loading](https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Elements/img),
[MDN picture fallback](https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Elements/picture).

---

## Evidence Boundaries

The page describes the demonstrated workflow and its reviewed evidence, not a
guarantee that every scientific route passes on every subject or platform.
Keep its release-state and review caveats until stronger evidence exists.
The film's metadata, audio-payload and compression checks complement visual and
audio review of the retained master.

Editable artwork, source renders and generation scripts are maintained in the independent [paper-latex repository](https://github.com/ChengjinLii/paper-latex). This software checkout retains reviewed website images and their SHA256 receipts.
