# SlicerDDSurfer project site: spec v2

Current-copy addendum, 26 October 2026: the hero and share title use the manuscript
title. `#idea` and its navigation entry are removed; the input layout moves to
Get started. Workflow cards follow six paper domains. The integration table now
lists implementation and runtime/access requirements instead of historical orange
partial-run badges; underlying validation records stay linked. The optional
reference-runtime row is not advertised. Get started expands the current GUI,
environment, input and DTI-first steps, with optional CLI/cohort details.
References [1]-[35] share one list inside the existing disclosure; the three
separate paper cards are removed. This addendum supersedes older copy/layout
instructions below, while keeping media integrity and publication boundaries.

Integration addendum, 22 October 2026 (revised 23 October 2026): the historical one-page design below is
retained for the overview. Measured performance evidence is now a section of the home page, `#performance`, reached
through one closed `<details>` ("Performance evidence"). It sits at the end of the page, after `#cite` and the visible
`#runtime` figure section, and opens by script when the URL hash is `#performance`. Runtime figures and performance
evidence are home-page sections only; the legacy redirect page was removed on 25 October 2026. The section reuses the existing design
and local fonts; no extra runtime service is required. `assets/data/performance.json` is the only allowlisted public
data export. Tables are generated from verified benchmark records into `index.html` (between the
`measured-evidence` markers); methods are collapsed by default. Full-module/ROI/substage scopes and
implementation/thread/scheduling/Python-port categories must stay distinct. The independent measurements do not change
the film's claims or provenance. See [README.md](README.md#performance-section) for current generation and checks.

Addendum, 23 October 2026: the home page also has a "What is integrated" status table (in `#workflows`), a route
guide figure (`#choose`), a run-time figure (`#runtime`) and a numbered reference list (`#cite`). §5.6.1, §5.6a,
§5.10, §5.10a and §5.10b below describe them. Their sources are listed in §1, rule 7.

Addendum 2, 23 October 2026: `#outputs` (§5.5) keeps its six film frames and adds nine tiles of the other methods
(free water, NODDI, DKI, MK-Curve, UKF, CSD/ACT, WMQL, TractCloud, SIFT2), rendered in the film's look from the
23 October 2026 runs on the same subject; `slicer.jpg` (§5.7) is a new full-window screenshot of the module. Their
sources are in §1, rule 7.

Addendum 3, 23 October 2026: the film is a new 72.0 s render (§2, §5.3, §7). It adds a Bundles scene (TractCloud on
the film run's DTI tractogram, WMQL on the multi-shell CSD/ACT run), the new module screenshot, and a fix for one-frame
black squares in the connectome scene. In `#outputs` the seven three-plane maps are renders of one view (the DEC film
frame's camera and planes, brighter plane frames), and the surfaces, tracts and connectome tiles are film moments
rendered at device scale 2 (§3).

Addendum 4, 24 October 2026: the home page copy is shorter (user request: less text, a shorter page). Sentences that
repeated another section are cut (the dataset and run story, the implementation origins, the “not certified” line in
the table caption); other sentences are shortened. No fact, honesty statement, section, table row, figure or image is
removed. §5 below holds the new copy; the integration table states its scope once, in the column header “Status on
Linux test subjects”, so its cells read “Supported”. The transcript, the reference entries, the paper cards and the
generated evidence are unchanged. The `claims_required` and `claims_banned` lists (§7.1 items 10 and 11) are
unchanged. Second pass, same day (layout, for a shorter page): the reference list [4] to [35] is in a closed
`<details>` (§5.10); the gallery has 4 columns on wide screens (§4); the integration table is compact (§5.6.1);
section padding and some inner margins are smaller (§4). Images, `<img>` attributes and the two figures are
unchanged. §7.2 item 8 checks the reference links.

Addendum 5, 24 October 2026: the seven slice maps in `#outputs` are new renders (§3; user request: the plane frame
lines were too bright). They show an axial and a coronal plane without frame lines, in one shared view. Their alt
texts say “on an axial and a coronal plane”. The route figure now has six example routes, and both figure PNGs have
new heights (route 2400×1713, run time 2400×1676).

Addendum 6, 24 October 2026: the film is a new 72.0 s render (§2, §3, §5.3, §7; user request: a more finished
look, no capsule labels, a Bundles scene without WMQL whose narration names no method). The type is a new system
of eyebrows, titles and notes; the Bundles scene shows TractCloud's bundles on the film run's DTI tractogram, and
line 10 is now “White-matter bundles, from the same tractogram.”; the two Slicer window moves (55.1–59.3 s) have
motion blur (180° shutter, sub-frames averaged in linear light). `poster.jpg`, `idea.jpg`, the three `out-*` tiles
and `social.jpg` are new frames of this render (§3 table); the note under the transcript names TractCloud as the
source of the bundles. The film's facts are in `FACTS.md` §9.

Addendum 7, 4 November 2026: the film is a new 64.8 s render (§2, §3, §7; user request: a shorter
introduction and the new wordmark at the end). The opening keeps only line 1 on diffusion MRI; line 2, the T1-weighted
card and the registration arrow are removed, so every later line, cue and section is 7.2 s earlier than in the 72.0 s
render (the opening camera now starts centred, so 4.8–7.0 s differs from the old shot). The last frames show only the wordmark and the last subtitle. The site copy is the CRF 24 encode of the
45,452,363-byte master (23,356,794 bytes, 1,944 frames, receipt in `published_assets.json`); `media/SlicerDSpace.en.srt`
has 12 cues. The frames in §3 were not taken again: `poster.jpg` and the `out-*` tiles show the same pictures, now
7.2 s earlier (poster 35.2 s); `idea.jpg` still shows the removed T1-weighted card and is not on the homepage.
`social.jpg` (the cover, with the old badge) and the README preview GIF (made from the 72.0 s film) are not changed
(user choice).
The §5.3 labels stay historical: the live play button reads “Play film” with no duration.

Date: 2026-10-20 (v2: fixes from the Opus spec review, round 1, listed in §9). Location: `docs/site/` (served) and `docs/website/` (spec and facts) on `main` of `ChengjinLii/SlicerDSpace`; the
original orphan `gh-pages` branch is kept as tag `archive/gh-pages-20261023`. Facts: [`FACTS.md`](FACTS.md) (cited
below as `[F<section>]`). The site makes no claim that is not in `FACTS.md`, except in the parts added on 23 October
2026 (§5.6.1, §5.6a, §5.10, §5.10a, §5.10b, and the performance section). Those parts have their own sources, named in
§1, rule 7.

## 1. Goal and limits

- A one-page project site for SlicerDDSurfer, with the promo film embedded and playable.
- Audience: diffusion MRI researchers who meet the project for the first time, and collaborators who will install it.
- The site must stay true while the repository is private. It states the pre-release status as the README does.
- Publishing is the owner's step: Settings → Pages → Source: GitHub Actions, then a manual run of the `Website`
  workflow (`.github/workflows/pages.yml`), which validates and deploys only `docs/site/` from `main`. Nothing in this
  repository enables Pages.

Settled decisions (not to be reopened in review):

1. Static HTML, CSS and a small vanilla JS file. No build step, no framework, no third-party requests at run time
   (fonts are self-hosted). Relative paths only, so the site works from any base path.
2. The served site is `docs/site/` on `main`. The site notes (`docs/website/`: `README.md`, `SPEC.md`, `FACTS.md`) and
   the site tests (`Testing/Website/`) are not served.
3. English only. Dark theme only (it matches the film). Phone to wide desktop.
4. The film file is the delivered film, byte for byte (no re-encode). Its `moov` box is already before `mdat`
   (faststart) [F9]. Subtitles are burned in, so the page has no `<track>`; it offers the `.srt` as a download and a
   transcript in the page.
5. No asset downloads (weights, templates, data) and no install-from-Extension-Manager claim for SlicerDDSurfer [F12.11].
6. Links into the repository are allowed; they need access while the repository is private. One note on the page says so.
7. Sources of the parts added on 23 October 2026. A row, a caption or a reference never claims more than its source:
   - Status table (§5.6.1): the code (`shared/orchestrator.py`, `shared/pipeline.py`, `microstructure/dki.py`,
     `microstructure/pipeline.py`, `tractography/*.py`, `connectome/pipeline.py`) and the dated records
     `docs/release/MODULE_RUNTIMES_20261017.md`, `docs/release/BUNDLE_WORKFLOW_20261021.md`,
     `docs/release/BUNDLE_BACKENDS_20261013.md`, `docs/release/SAF_PYTHON_20261013.md`,
     `docs/release/ALGORITHM_SELECTION_20261021.md`, `docs/setup/SUPPORT_MATRIX.md`, `docs/setup/UKF.md`,
     `docs/setup/MICROSTRUCTURE.md` and `docs/architecture/IMPLEMENTATION_PROVENANCE.md`.
   - Figures (§5.6a, §5.10a): `paper-latex:docs/figures/README.md` (legends, generators and data files) and the data files it names
     (`docs/release/benchmarks/module_runtimes_20261017.json`, `docs/site/assets/data/performance.json`).
   - Performance section (§5.10b): `docs/release/benchmarks/*.json`, through `scripts/build_performance_site.py`.
   - References (§5.10): the DOI record of each entry (Crossref or DataCite) and the page of the tool that asks for the
     citation. Entries that could not be checked are left out.
   - Result tiles `res-*.jpg` (§5.5) and `slicer.jpg` (§5.7): the pipeline outputs and QC files of the runs, through
     `P/tools/export_gallery.py` (each source output and its SHA-256 in `P/film/data/gallery/index.json`),
     `P/film/gallery.js` (stills) and `P/tools/make_site_tiles.py` (crop and resize; `P/renders/site_tiles/tiles.json`),
     and `P/runtime/gui_workflow_capture.py` for the screenshot. A tile shows only products that the pipeline published
     (not withheld as failed); a caption states any automatic QC result that did not pass.

## 2. Files

```
docs/website/README.md                        owner notes: what the folder is, how to publish, how to test, lines to revisit at public release
docs/website/SPEC.md                          this file
docs/website/FACTS.md                         fact sheet
Testing/Website/check_site.py                 static checks (Python 3 stdlib; ffprobe optional)
Testing/Website/browser_check.mjs             headless Chrome checks (playwright-core)
docs/site/.nojekyll
docs/site/index.html
docs/site/404.html                            small page, same header/footer style, link home
docs/site/assets/css/site.css
docs/site/assets/css/performance.css          styles of the performance evidence inside `#performance`
docs/site/assets/css/guide-bot.css            styles of the project guide (mascot and dialog)
docs/site/assets/js/site.js
docs/site/assets/js/performance.js            filters and method links of the performance evidence
docs/site/assets/js/guide-bot.js              local project guide (curated answers; no network request, storage or model service)
docs/site/assets/data/performance.json        measurement catalogue of the performance evidence (the only allowlisted public data file)
docs/site/assets/fonts/                       InterTight, Inter, IBMPlexMono as .woff2 (Latin subset) + the three OFL .txt files
docs/site/assets/img/                         icon-64.png, icon-180.png (from R/Resources/Icons/SlicerDDSurfer.png), logo-ddsurfer.png
                                              (header and footer logo), guide-bot-hat.svg (project-guide mascot), poster.jpg,
                                              social.jpg, idea.jpg, out-surfaces.jpg, out-tracts.jpg, out-connectome.jpg, slicer.jpg,
                                              fig-product-map.png, fig-route-guide.png, fig-runtime-resources.png, fig-runtime-optimizations.png (the four
                                              figures are rasterised from `paper-latex:docs/figures/*.svg`, which is not served), and twelve result tiles: res-dec.jpg,
                                              res-fw.jpg, res-ndi.jpg, res-mk.jpg, res-mkcurve.jpg, res-tissue.jpg, res-parcel.jpg, res-tr-ukf.jpg,
                                              res-tr-csd.jpg, res-wmql-ukf.jpg, res-tc-ukf.jpg, res-net-sift2.jpg
docs/site/media/SlicerDSpace_promo_1080p.mp4  CRF 24 encode of P/out/SlicerDSpace_promo_1080p.mp4 (23,356,794 bytes; Addendum 7)
docs/site/media/SlicerDSpace.en.srt           copy of P/out/SlicerDSpace.en.srt
```

## 3. Images (film frames, the cover, the two figures of §5.6a and §5.10a, twelve result tiles and the screenshot)

`poster.jpg` and `idea.jpg` come from `P/out/SlicerDDSurfer_promo_1080p.mp4` with `ffmpeg -ss <t> -i <film> -frames:v 1`.
The three `out-*` tiles are the same film moments rendered from the film page at device scale 2 (3840×2160,
`P/tools/film_frames_hd.mjs`; the page draws its 3D canvas at that size), cropped with the boxes below × 2 and
resized (JPEG quality 90), so they are not enlarged from the encoded video. Crop boxes are `x0,y0,x1,y1` in the
1920×1080 frame. A crop taken while a subtitle cue is on screen ends at y ≤ 945 (subtitle text starts near y = 950;
two-line cues reach y ≈ 1040). Frames at 41.6, 42.4 and 54.8 s show no cue. Check each output by eye; adjust a box by up to 60 px if a brain or label is cut, and record the final
boxes in `README.md`. JPEG quality 82, progressive, sRGB.

| File | t (s) | Crop box | Output size | Shows |
|---|---|---|---|---|
| poster.jpg | 42.4 | full frame | 1920×1080 | whole-brain DTI tractography in the glass pial surface (no subtitle at 42.4 s) |
| idea.jpg | 9.0 | 60,60,1760,945 | width 1400 | DWI slice, "registration" arrow, "+ T1-weighted scan" card |
| out-surfaces.jpg | 35.3 | 684,220,1644,940 | 1200×900 | DDSurfer pial surface |
| out-tracts.jpg | 41.6 | 520,228,1656,1080 | 1200×900 | tractography |
| out-connectome.jpg | 54.8 | 548,68,1860,1052 | 1200×900 | connectome nodes/edges and matrix |
| slicer.jpg | — | `P/runtime/shots_workflow_v2/results_window.png` (2816×2054), resized | 1408×1027 | real 3D Slicer screenshot (§5.7) |
| social.jpg | — | from P/out/cover_1280x640.png | 1280×640 | the film's cover (og:image) |

Every `<img>` has `alt` (the “Shows” text above, the tile's alt text in §5.5, or the icon’s name), `width`, `height`
and `loading="lazy"`. (The poster is the `<video poster>`, not an `<img>`.)

Result tiles (`res-*.jpg`, 1200×900, progressive), rendered in the film's look by `P/film/gallery.js` and cropped
and resized by `P/tools/make_site_tiles.py` (sources and SHA-256 in `P/renders/site_tiles/tiles.json`):
- the seven slice maps (`res-dec`, `res-fw`, `res-ndi`, `res-mk`, `res-mkcurve`, `res-tissue`, `res-parcel`):
  3840×2160 stills in the film's look (`?look=film`) with one shared view: an axial and a coronal plane, the sagittal
  plane hidden, no plane frame lines (`frame=0&hide=0&cam=235,38,470,0,4,0,30,0.1,-0.02`; the full query is in
  `tiles.json` `view`), crop 1398,440,3202,1794 (the union box of the seven brains), resized to 1200×900 with Lanczos. The inside of the brain is as sharp as the
  exported volumes allow (about 280 voxels across, trilinear sampling); no sharpening is applied;
- tracts, bundles and networks: 1600×1200 stills, crop 50,37,1550,1162, JPEG quality 84.
Nothing is drawn on the data (the slice maps have no plane frames since Addendum 5). The
screenshot shows the subject folder through a short link (`/Users/Shared/EDDEN/sub-01` → the run's subject folder),
with the log cleared by its own Clear button and the Reason column widened, so no path of the build machine is in view.

## 4. Look

- Tokens (CSS custom properties on `:root`):
  `--bg #06080b; --bg-2 #0a0e13; --panel #0e131a; --line rgba(236,232,223,.09); --line-2 rgba(236,232,223,.16);
  --text #ece8df; --muted #9ba5b0; --red #ff5a52; --green #46d38a; --blue #5b8cff; --orange #f0913f; --ivory #efe6d2;
  --csf #4d86e8; --radius 14px; --max 1200px; --gutter clamp(16px, 4vw, 40px)`.
- Type: Inter Tight (headings, 600, letter-spacing −0.02 to −0.035em), Inter (body, 16–18px, line-height 1.6),
  IBM Plex Mono (eyebrows, labels, chips, code; 12–13px, uppercase only for eyebrows, letter-spacing .08em).
  H1 `clamp(40px, 6.2vw, 84px)`; H2 `clamp(30px, 3.8vw, 52px)`.
- Motif: the DEC colours. A 2px "tri-rule" (`linear-gradient(90deg, var(--red), var(--green), var(--blue))`, 70%
  opacity) under the header and above each section eyebrow. Chips look like the film's labels: mono text,
  1px `--line-2` border, 6px radius, the product name in `--text` and the qualifier in `--muted`
  (e.g. `DDSeg · tissue`).
- Background: `--bg` with two fixed soft radial glows (blue upper left, faint green lower right, ≤ 10% opacity).
- Panels: `--panel`, 1px `--line`, `--radius`. No drop shadows except a soft DEC glow behind the film frame.
- Motion: sections fade and rise 12px once when they enter the viewport (IntersectionObserver, 600ms). None under
  `prefers-reduced-motion: reduce`. Content must be visible without JS (add the hidden state only from JS).
- Layout: max width `--max`, side gutter `--gutter`. Section padding `clamp(44px, 5.5vw, 76px)` (Addendum 4).
  Grids collapse: gallery 4 → 3 (≤ 1099px) → 2 (≤ 699px; smaller tile text on phones), tiles stay 4:3;
  two-column sections → 1 (≤ 900px); workflow cards 5 → 2 → 1. No horizontal page scroll at 320px or wider; code
  blocks scroll inside themselves (`overflow-x: auto`), grid children have `min-width: 0`.
- Header: sticky, 64px, `--bg` at 80% with backdrop blur, bottom tri-rule. Left: icon (28px) + "SlicerDDSurfer".
  Centre/right: anchor links. Right: chip "Private pre-release". At ≤ 1023px the links fold into a "Menu" button
  (`aria-expanded`, keyboard operable, closes on link click and Escape).
- Accessibility: skip link to `#main`; visible `:focus-visible` outline (2px `--blue`, 2px offset); one `<h1>`;
  headings in order; landmarks (`header`, `nav`, `main`, `footer`); body text contrast ≥ 7:1, small text never below
  `--muted`.

## 5. Page content (exact copy)

Typography marks: use the real characters “ ” ’ – — · › × in the HTML. Text in quotes below is the copy. `[F…]` marks
are sources for review and do not appear on the page.

### 5.0 Head

- `<html lang="en">`, `<meta charset="utf-8">`, viewport `width=device-width, initial-scale=1`.
- Title: “SlicerDDSurfer — Brain anatomy from diffusion MRI, inside 3D Slicer”.
- Description: “SlicerDDSurfer brings diffusion-MRI anatomical reconstruction, microstructural mapping and white-matter
  connectivity analysis into one 3D Slicer module.” [F1 README:15-18]
- `theme-color` `#06080b`; `color-scheme: dark`.
- Open Graph / Twitter: `og:type website`, `og:title` = the title, `og:description` = the description,
  `og:url https://chengjinlii.github.io/SlicerDDSurfer/`, `og:image
  https://chengjinlii.github.io/SlicerDDSurfer/assets/img/social.jpg` (1280×640), `twitter:card summary_large_image`.
  `<link rel="canonical">` to the same `og:url`. (These two absolute URLs are the expected Pages address; README lists
  them under “revisit”.)
- Icons: `assets/img/icon-64.png` (`rel=icon`), `assets/img/icon-180.png` (`rel=apple-touch-icon`).
- Fonts: `@font-face` with `font-display: swap`; preload the Inter Tight and Inter files.

### 5.1 Header nav

Links: “Film” `#film`, “Idea” `#idea`, “Outputs” `#outputs`, “Workflows” `#workflows`, “In Slicer” `#slicer`,
“Status” `#status`, “Get started” `#start`, “Cite” `#cite`. Chip: “Private pre-release” [F1 README:20-22].

### 5.2 Hero (`#top`)

- Eyebrow: “3D Slicer module · Diffusion MRI”.
- H1: “Brain anatomy from diffusion MRI, inside 3D Slicer.” [F9 narration 12; F1 README:15-18]
- Lead: “SlicerDDSurfer builds tissue labels, brain regions and cortical surfaces from diffusion MRI, with no
  T1-weighted scan at analysis time. It also maps microstructure, traces white-matter pathways and builds structural
  connectomes.” [F2 FACTS:10; F3 README table]
- Buttons: primary “Watch the film” (scrolls to the `#film` section and calls `play()` on `#film-video`), secondary “Get started” (`#start`).
- Status line under the buttons (small, `--muted`): “Research software · Private pre-release · BSD-3-Clause code”
  [F1 README:20; LICENSE].

### 5.3 Film (`<section id="film">`, with a visually hidden `<h2>Film</h2>`)

- A wide frame (max 1200px, 16:9, `--radius`, 1px `--line-2`, soft DEC glow) holding:
  `<video id="film-video" controls playsinline preload="metadata" poster="assets/img/poster.jpg" width="1920"
  height="1080" aria-label="SlicerDSpace film, 72 seconds, narrated, with subtitles in the picture">` with
  `<source src="media/SlicerDSpace_promo_1080p.mp4" type="video/mp4">` and a fallback link “Download the film (MP4)”.
- Progressive enhancement: JS removes `controls` at start and shows a centred play button (round, 88px, glass,
  triangle icon, label “Play film · 1:12”). Click/Enter/Space: hide the button, add `controls`, `play()` with sound.
  On `ended`: show the button again (label “Play again”). Without JS the native controls stay.
- Meta row under the frame (mono, small): “72.0 s · 1920 × 1080 · narrated · subtitles in the picture” and links
  “Download MP4 (48 MB)” `media/SlicerDSpace_promo_1080p.mp4` (`download`), “Subtitles (.srt)”
  `media/SlicerDSpace.en.srt` (`download`). [F9]
- `<details>` “Transcript”: an ordered list of the 13 narration lines, verbatim from the `.srt`, in order [F9]:
  1. Diffusion MRI records how water moves through the living brain.
  2. Mapping anatomy has usually meant a second, T1-weighted scan, registered into diffusion space.
  3. SlicerDDSurfer reconstructs anatomy directly from diffusion MRI, without a T1-weighted scan.
  4. It maps microstructure, from DTI to free water, NODDI and kurtosis.
  5. Tissue labels.
  6. Brain regions.
  7. White and pial surfaces,
  8. all in native diffusion space.
  9. Whole-brain tractography, traced in the same native space.
  10. White-matter bundles, from the same tractogram.
  11. And structural connectomes, built on the same anatomy.
  12. One 3D Slicer module, with automatic QC and a record of every run.
  13. SlicerDDSurfer. Brain anatomy from diffusion MRI, inside 3D Slicer.
- Note under the transcript (small): “Most scenes come from one run on EDDEN sub-01 (b = 0 and b ≈ 1000 volumes). The bundles are TractCloud’s, on that run’s DTI tractogram; the 3D Slicer screenshot comes from multi-shell runs. The film names free water, NODDI and kurtosis but does not show them; Outputs below does.” (The dataset is named in full in §5.5; the date of the later runs is in the footer, §5.11.) [F9; F12.15; Addendum 3]

### 5.4 The idea (`#idea`)

Two columns: text left, `idea.jpg` right.

- Eyebrow: “The idea”.
- H2: “Anatomy from the diffusion scan itself.”
- P1: “Diffusion studies have usually needed a second, T1-weighted scan, registered into diffusion space.”
  [F2 FACTS:9]
- P2: “SlicerDDSurfer works from the diffusion scan. DDSeg labels tissue, DDParcel labels regions and DDSurfer builds
  white and pial surfaces, all in native diffusion space.” [F2; F3; F9 narration 5–8]
- Input card (mono, panel), title “Canonical input”:
  ```
  sub-01/
    input/
      data.nii.gz
      bvals
      bvecs
      nodif_brain_mask.nii.gz   # optional; generation needs a supported backend
  ```
  caption: “DWI and mask must share one voxel grid.” (The layout above already marks the mask as optional.)
  [F6 README:100-113]
- Footnote (small, `--muted`): “‘No T1-weighted scan’ describes analysis time. The models were trained with labels
  derived from T1-weighted MRI.” [F2 FACTS:10]
- Image caption: “From the film: the usual route.” (P1 says what the usual route is.)

### 5.5 Outputs (`#outputs`)

- Eyebrow: “One subject, one space”.
- H2: “From one diffusion dataset to seven kinds of result.”
- Intro: “Data: EDDEN sub-01 (OpenNeuro ds004666, CC0), AVG_mag series. The series averages five repeated acquisitions, so its SNR is higher than a typical single scan. Six images come from the film’s run (b = 0 and b ≈ 1000; 117 of 297 volumes). The others come from runs of 23 October 2026 on the same Mac: UKF on the same input; the multi-shell methods on b = 0, 1000 and 2000 (207 of 297 volumes).” [F9 Data; P/MAKING_OF.md:69-72, 142; P/tools/make_subject_ms.py; P/runtime/r2.sh]
- Fifteen tiles (image, chip, title, caption), in this order:

| Image | Chip | Title | Caption | Source |
|---|---|---|---|---|
| res-dec.jpg | “DTI · microstructure” | “Direction-encoded colour” | “Principal diffusion direction weighted by FA, from the DTI fit.” | F9 caveats (DEC = \|e1\| × FA) |
| res-fw.jpg | “Free water · microstructure” | “Free-water fraction” | “Free-water tensor model (DIPY) on b = 0, 1000 and 2000.” | P/film/data/gallery/index.json; QC file of the run |
| res-ndi.jpg | “NODDI · microstructure” | “Neurite density” | “NDI from the NODDI fit (AMICO) on two shells.” | P/film/data/gallery/index.json; QC file of the run |
| res-mk.jpg | “DKI · microstructure” | “Mean kurtosis” | “MK from the DKI fit (DIPY, WLS).” | P/film/data/gallery/index.json; QC file of the run |
| res-mkcurve.jpg | “MK-Curve · microstructure” | “Mean kurtosis, MK-Curve” | “Unofficial Python port, same data. Automatic QC did not pass: 98.7% of mask voxels are valid; the check needs all.” | P/film/data/gallery/index.json; QC file of the run |
| res-tissue.jpg | “DDSeg · tissue” | “Tissue labels” | “White matter, grey matter and CSF, predicted from DTI features.” | F4 (five DTI features) |
| res-parcel.jpg | “DDParcel · regions” | “Brain regions” | “Anatomical regions labelled in the diffusion volume.” | F7 (DDParcel title) |
| out-surfaces.jpg | “DDSurfer · surfaces” | “Cortical surfaces” | “White and pial meshes in native diffusion space. Shown: the pial surface.” | F2 FACTS:13; the 35.3 s frame shows the pial surface (P/film/main.js:350, white → pial morph ends at 35.2 s) |
| out-tracts.jpg | “Tractography · DTI” | “Whole-brain tractography” | “Deterministic DTI tractography. Shown: a display subset of 24,000 of 130,780 streamlines.” | F9 caveats |
| res-tr-ukf.jpg | “Tractography · UKF” | “UKF two-tensor tractography” | “UKFTractography. Shown: a display subset of 24,000 of 208,295 streamlines.” | P/film/data/gallery/index.json; QC file of the run |
| res-tr-csd.jpg | “Tractography · CSD/ACT” | “MSMT-CSD with ACT” | “MRtrix3 iFOD2 with ACT. Shown: 24,000 of 1,000,000 streamlines.” | P/film/data/gallery/index.json; QC file of the run |
| res-wmql-ukf.jpg | “WMQL · fiber bundles” | “Multi-bundle query results” | “Official WMQL queries on UKF tractography. Colours distinguish query families; a display subset is shown.” | unchanged copy of `paper-latex:docs/figures/src/results/bu_wmql_ukf.jpg`; source and hash in `paper-latex:docs/figures/src/results/wmql_sources.json` (`scripts/check_website_assets.py --check`) |
| res-tc-ukf.jpg | “TractCloud · bundles” | “42 white-matter tracts” | “TractCloud on the UKF tractogram: 84,356 streamlines classified; 9,321 shown.” | P/film/data/gallery/index.json; QC file of the run |
| out-connectome.jpg | “Connectome · DDParcel nodes” | “Structural connectome” | “The 260 strongest of 854 edges are drawn; the matrix shows all 84 × 84 nodes.” | F9 caveats |
| res-net-sift2.jpg | “Connectome · SIFT2” | “SIFT2-weighted connectome” | “From the CSD/ACT tractogram. The 260 strongest of 2,726 edges are drawn.” | P/film/data/gallery/index.json; QC file of the run |

- Note under the grid (small): “All images use the film’s look; colours, subsets and upsampling change the look, not the data. Automatic QC passed for every stage shown that records it, except MK-Curve. WMA also ran on the UKF tractogram but is not shown: it failed the pipeline’s sampling check (right arcuate fasciculus: 6 streamlines, minimum 20). No expert has reviewed these outputs.” (“Every stage shown that records it” includes the DTI, DDSeg, DDParcel and DDSurfer stages.)
  [F9: MAKING_OF:26-33, 76-78; the QC files of each run; `microstructure/mkcurve.py:265`;
  P/runtime/subjects_ukf/sub-01/outputs/04_tractography/bundles/wma_org_atlas/ukf/wma_org_atlas_qc.json]
- The UKF run used the film's input with 21 mask voxels removed where the mean b = 0 signal is zero (P/runtime/r2.sh).

### 5.6 Workflows (`#workflows`)

- Eyebrow: “Workflows”.
- H2: “One module, five workflow families.” [F3 README:28]
- Intro: “Select only the products you need, and install dependencies only for their routes.” [F3 README:38]
- Five cards, numbered 01–05. Each: title, “Products” list, “Needs” line. Text from the README table [F3], except as
  noted below the list:
  1. “Input preparation” — Products: “DICOM/NIfTI/NRRD import, gradient validation, optional mask generation and
     preprocessing” — Needs: “Appropriate import/preprocessing backend”.
  2. “Anatomical reconstruction” — “DDSeg tissue labels, DDParcel regions, DDSurfer white/pial surfaces” —
     “Matching model assets, PyTorch and SlicerDMRI for the selected route”.
  3. “Microstructural analysis” — “DTI FA/MD/AD/RD, optional Free-water, NODDI and DKI, regional summaries” —
     “Route-specific packages and acquisition suitability checks”.
  4. “White-matter pathways” — “DTI/CSD tractography, ACT/SIFT2, partial UKF and SAF routes, WMQL/WMA/TractCloud bundles” —
     “Backend-specific tools and assets”.
  5. “Structural connectivity” — “Regional matrices (streamline count; SIFT2-weighted for CSD/ACT), Slicer scene
     inspection, QC and provenance” — “Compatible parcellation and tractogram; SIFT2 weights for the weighted matrix”.
  (Cards 4 and 5 differ from the README table on purpose: UKF and SAF are partial, and the GUI and CLI reach only the
  streamline-count and SIFT2-weighted matrices. That scalar and length weights are Python API only is stated once, in
  the table row of §5.6.1.)
- Note row (three short items with a small icon or chip each) [F3]:
  - “Microstructure runs DTI only by default; turn on Free-water, NODDI or DKI under Advanced.”
  - “MRtrix3, FSL, FreeSurfer and advanced bundle backends are optional and are not installed silently.”
  - “A missing backend is reported, never silently replaced.”

#### 5.6.1 What is integrated (inside `#workflows`, after the note row)

- H3 “What is integrated”, then a table in a scroll wrapper (`.table-wrap`, `tabindex="0"`, `role="region"`; the table
  has `min-width: 980px`, so it scrolls inside the wrapper on small screens and the page never scrolls sideways).
- Caption: “A status describes this project’s integration and its dated records, not the method itself. Expert review
  of the outputs is pending.” (That macOS and Windows scientific workflows are not certified is stated in §5.8.)
- Columns: Stage · Method · How it is integrated · Status on Linux test subjects · Reference [n]. One row per method,
  26 rows.
- Status values: “Supported” (the column header gives the scope, Linux test subjects; the WMA row reads “Supported
  with UKF input”); “Partial: <one-line reason>”; “Needs <tool>” as a second line where an external tool is required
  (FreeSurfer, MATLAB, MRtrix3, FSL, dcm2niix or DWIConvert, UKF extension). A row that is both partial and
  tool-dependent shows both lines; Partial is never dropped.
- The MK-Curve (Python) row says “Same plan check as DKI” instead of repeating the DKI row's shell, direction, volume
  and rank rule.
- Partial rows: MK-Curve (MATLAB reference), UKF two-tensor, SAF, WMQL on sparse input, WMA on CSD input, TractCloud on
  CSD input, scalar and length connectome weights (Python API only).
- Reference cells link to `#ref-<n>` in the reference list (§5.10).
- Compact layout (Addendum 4): cell padding 7px 10px, line height 1.4, 13.5px text. The reference cell wraps between
  `[n]` links (each link stays whole), so the “How it is integrated” column gets the width. All 26 rows and every
  Partial and Needs line stay visible.

### 5.6a Which route when (`#choose`, after `#workflows`)

- Eyebrow “Choosing routes”, H2 “Which route when.”, then two panels, `fig-product-map.png` and below it
  `fig-route-guide.png`, each wrapped in a link to the full-size file (`target="_blank" rel="noopener"`), with `width`,
  `height`, `loading="lazy"`, `decoding="async"`.
- Captions: “What each method makes and what it reads. Free water, NODDI and DKI need at least two non-zero shells;
  DKI counts only shells up to 2500 s/mm².” and “Six example routes that together use every method. Built from the
  planner and method code and the documented dependencies; status is recorded for Linux test subjects.” No caption
  may say that a single-shell scan can use CSD/ACT.

### 5.7 In Slicer (`#slicer`)

- Eyebrow: “In 3D Slicer”.
- H2: “One 3D Slicer module, with QC and a record of every run.” [F9 narration 11; F1 README:15-18]
- `slicer.jpg` (1408×1027) full width in a panel. Caption: “A real screenshot of 3D Slicer 5.12.4 on macOS. The module loaded the multi-shell EDDEN results: DDParcel labels over free-water-corrected FA in the slice views, the run’s recorded QC in Results and QC. The 3D view shows all 42 TractCloud bundles: the module loads four, and the capture script added the others with Slicer’s own reader. The script also set the inputs, the products and the view settings; the runs were started from the command line.”
  [P/runtime/capture_gui_workflow.sh; P/runtime/gui_workflow_capture.py; P/runtime/shots_workflow_v2/capture.log]
- Ordered list “A run in six steps” [F6 README:89-94]:
  1. “Open SlicerDDSurfer and choose English or Chinese.”
  2. “Enter the Subject ID, the subject root and, optionally, an output directory.”
  3. “Tick only the products you need.”
  4. “Choose each product’s algorithm. Prepare dependencies in Environment; set detailed parameters in Advanced
     settings.”
  5. “Click Run SlicerDDSurfer. Missing inputs turn red; the log says what to correct.”
  6. “Key results of successful stages load into the scene. Review QC, reload, replace or clear them in Advanced
     settings › Results and QC.”
- Side note (panel, mono command) [F3 README:144-152]: title “From the command line”, text “Check inputs and
  environment without running anything:”, code
  `PythonSlicer run_subject_pipeline.py --subject-root /path/to/sub-01 --targets validation,microstructure --microstructure-stages dti --doctor`,
  then “Use `--plan` instead of `--doctor` to write a run plan.”

### 5.8 Status (`#status`)

- Eyebrow: “Status”.
- H2: “Research software, stated plainly.”
- Three columns, each a panel with a coloured top edge (green, orange, blue):
  - “Recorded” (green): “Linux real-data and GUI records for specific checkouts, routes and datasets.” / “A UI
    validation record of the compact interface.” [F5 README:158-163]
  - “Not certified” (orange): “Native Windows and macOS scientific workflows.” / “Three-platform SDK builds.”
    [F5 README:174]
  - “Not yet released” (blue): “Private pre-release; not in the official Extensions Catalog.” / “The Extension Index
    JSON is a draft Tier 1 proposal: not submitted, and not acceptance by Slicer.” / “Public source access,
    third-party licence clearance and asset redistribution review remain release gates.” [F1; F5 README:175-176;
    EXTENSION_READINESS.md:106-107]
- Block “Implementation origins” [F4 README:42-45; PROVENANCE:25]: “The integration table under Workflows says how
  each method is implemented: reimplemented, ported, bundled or called.” (“integration table” links to `#integrated`;
  the table rows for DDSeg, SAF, MK-Curve, PNL CNN and DDSurfer carry the origin of each.)
  Link: “Implementation provenance and verification limits” → repo `docs/architecture/IMPLEMENTATION_PROVENANCE.md`.
- Callout (full width, panel, larger text): “A completed run is not a clinical assessment or proof of scientific
  correctness.” [F1 README:21-22]
- `<details id="performance-notes">` “Performance & implementation notes”: “Measured changes, equivalence checks
  and limits, with each scope kept separate.” and the link “Explore the performance evidence” → `#performance`.

### 5.9 Get started (`#start`)

- Eyebrow: “Get started”.
- H2: “Install from source.”
- Callout (top): “The repository is private during pre-release; you need access to clone it or open its links.”
  [F1 README:51] (Not the model-asset repository: `SlicerDDSurfer-models` is public,
  R/docs/release/VALIDATION_20260816.md:36.)
- Steps, each with a short title, text and (where given) a code block with a “Copy” button [F6 README:51-81]:
  1. “Stage the extension” —
     ```
     git clone https://github.com/ChengjinLii/SlicerDDSurfer.git
     cd SlicerDDSurfer
     PythonSlicer scripts/stage_slicer_extension.py \
       --install-root ~/SlicerDDSurfer-extension --link-mode copy --clean
     ```
     “Set `--slicer-version` to your Slicer version. This is a source install, not an official CPack release.”
  2. “Add the module path” — “In Edit › Application Settings › Modules, add the expanded absolute path of
     `~/SlicerDDSurfer-extension/SlicerDDSurfer/lib/Slicer-5.10/qt-scripted-modules` to Additional module paths and
     restart Slicer. GUI validation uses Slicer 5.10.”
  3. “Add the extensions your routes need” — “Install SlicerDMRI and PyTorch from Slicer’s Extension Manager.”
  4. “Get the model files” — “Read the asset terms, then download only the profile you need to a writable folder
     outside the checkout:”
     ```
     PythonSlicer scripts/download_model_assets.py --profile surface \
       --asset-root ~/SlicerDDSurfer-model-assets --source url \
       --source-id slicerdsurfer_models_git
     ```
     “Availability and SHA256 verification do not establish redistribution rights.”
  5. “Prepare the input” — “Use the canonical layout under Idea. Gradients must match the DWI; motion-corrected DWI
     needs corrected gradients.” (“Idea” links to `#idea`.)
- Doc links (chips/buttons, `target="_blank" rel="noopener"`): “Quick Start” → `docs/setup/QUICKSTART.md`,
  “Support matrix” → `docs/setup/SUPPORT_MATRIX.md`, “Model asset terms” → `docs/assets/MODEL_ASSET_TERMS.md`,
  “Repository” → repo root. All under `https://github.com/ChengjinLii/SlicerDDSurfer/blob/main/` (root: the repo URL).

### 5.10 Cite and references (`#cite`)

- Eyebrow “Cite”, H2 “Cite the software and the methods you use.”
- Text: “The protocol manuscript is in preparation. Cite the software version you used (CITATION.cff) and the methods
  you ran.” Link “CITATION.cff” → repo `blob/main/CITATION.cff`. [F7]
- Three paper cards for DDSurfer, DDSeg and DDParcel, exactly as CITATION.cff [F7], numbered [1] to [3]
  (`id="ref-1"` to `ref-3`).
- H3 “References”, then the intro “The Reference column of the integration table points here. [1] to [3] are the
  papers above; [34] and [35] are the film’s data.” (“integration table” links to `#integrated`), then a closed
  `<details id="reference-list" class="transcript ref-details">` with the summary “Reference list [4] to [35]”. It
  holds one numbered list, [4] to [35], in groups (H4): Microstructure; Tractography and connectomes; Bundles; Platform
  and tools; Data. A click on a `#ref-<n>` link opens it (`site.js` opens any closed `<details>` that holds a hash
  target); without JS, Chrome opens it on fragment navigation. Each item: lead author (“Surname Initials, et al.”; both names for two authors), title,
  venue, volume(issue):pages, year, and a DOI link `https://doi.org/<doi>`. Items without a DOI (ISMRM abstracts) link to
  the ISMRM archive page; the 2016 Dhollander workshop abstract has no link. Items marked unverified in the review notes
  are left out (for example the LNCS volume of TractCloud). Every external URL is in the allowed set (§6).
- Left out on purpose: SlicerDMRI (Norton 2017) and FastSurfer, because their titles contain “Open Source” and
  “accurate”, which `claims_banned` rejects in the site's text; optional entries the review notes did not tie to code
  (Reddy and Rathi 2016, Destrieux 2010, PyTorch).

### 5.10a Run times (`#runtime`, directly before `#performance`)

- Eyebrow “Run times”, H2 “Recorded run times, per module.”, then separate
  `fig-runtime-resources.png` and `fig-runtime-optimizations.png` figures, same markup as
  §5.6a (the width and height attributes equal the pixel size of the file).
- Stage-time caption: scoped log-scale observations, seven same-input CPU/CUDA routes,
  one baseline per algorithm and aligned CPU/CUDA value columns, shared host and different thread budgets, no summed
  workflow total or controlled device speed ratios. Hardware in this figure's title
  comes from actual device repeats, not an inferred report-generation host. Other
  CPU-only rows keep their dated scopes. SAF re-filters 100,000 existing tracks;
  TractCloud uses 50,000 tracks.
- Optimization figure: ten timing pairs above a matching-ID, three-column table
  (ID, computation, optimization), one line per method. Original figure typography
  is retained: 7-point regular labels, 6.5-point values, 6-point auxiliary text,
  inline muted scope and ample row padding. Lines join only the timing points,
  not the left axis. Equivalence checks stay in the source records. Groups are complete model calls,
  local computations, and reuse/data movement. Thread counts, device switches,
  scheduling and prior-plugin-only integration fixes are not included. MK-Curve
  is port-only, surface mapping credits upstream DDSurfer, and DWI is I/O only.
  Captions preserve scope; overlapping gains are not added into a workflow speedup.

### 5.10b Performance (`#performance`, after `#runtime`)

- Eyebrow “Performance”, H2 “Measured changes and their limits.”, then one `<details id="performance-evidence"
  class="transcript perf-evidence">`, closed by default, with the summary “Performance evidence”.
- Inside: the intro and the two disclaimers, the filter buttons (shown by script only), the generated evidence region
  (charts, table, note, method records; written by `scripts/build_performance_site.py`), the four “levers” and the
  “boundary of the evidence” list. The headline of the old subpage (“Same methods. Less overhead.”) is not kept.
- The catalogue contains 13 traceable measurements as of 25 October 2026, including
  the preserved thread/scheduler/API-integration controls omitted from the figure.
  Source-freshness review remains dated 22 October, not relabeled as a new audit.
- Hand-written copy around the generated region (shortened 24 October 2026; the generated method records carry the
  details):
  - Lead: “Remove repeated work, keep the computation, measure only what was tested.”
  - Disclaimers: “No whole-workflow speedup has been established. Output equivalence is not anatomical validation or
    an increase in predictive validity.” / “Upstream versions rechecked on 22 October: DIPY 1.12.1, AMICO 2.1.1,
    SciPy 1.18.1. Each method record separates current-source checks, older timings and upstream capabilities.”
  - “Measured work” / “Time, with context.” / “Full models, small regions and single operations are different
    experiments; they stay separate.”
  - Four levers (“What the numbers mean” / “Four different levers.” / “A larger ratio does not make two experiments
    comparable.”): Implementation, “Do the same work once.”, “Batch equivalent operations. Estimator,
    initialization, solver, precision and QC thresholds stay the same.” · Native threads, “Use existing compute
    capacity.”, “AMICO’s own threads, not a new NODDI algorithm. Hardware is part of the measurement.” · Scheduling,
    “Overlap independent jobs.”, “FW, NODDI and DKI overlap once their inputs are ready. Default concurrency is one.” ·
    Python-port control, “Name the real reference.”, “MK-Curve port changes are compared with earlier Python code, not
    the author’s MATLAB.”
  - “Still open” / “The boundary of the evidence.” / “These measurements cover specific Linux inputs and settings.
    They do not replace independent scientific review.”, the link “Read the complete optimization report”, and “The
    JSON is a limited public summary, not private logs or subject data.” (The private-repository note is the one in
    §5.9.)
  - Open items: Whole workflow, “No matched end-to-end comparison. Do not add savings or multiply ratios across
    overlapping stages.” · Generalization, “Single pairs, no confidence interval; more subjects and hardware are
    needed.” · Long-task coverage, “A full-brain DKI pair, later default-QR MK-Curve timing and a matched
    CSD/ACT/SIFT2 comparison remain open.” · Other methods, “No run-time gain over the authors’ code is claimed for
    DDSeg, DDParcel, SAF, UKF or TractCloud. DDSurfer surface mapping is upstream work.” · Validity and platforms,
    “WMA small-bundle sampling and invalid fit voxels need their own review. Native Windows/macOS scientific timing is
    not certified.” · Next experiment, “A persistent-model worker that reuses weights across subjects is proposed, not
    built or measured.”
- Without JavaScript the native `<details>` still opens and shows the table. With JavaScript, `site.js` opens the
  details when the hash is `#performance` (on load, on `hashchange` and on a click on a link to it) and opens any closed
  `<details>` that contains the target of a hash (for example `#method-freewater`).

### 5.11 Footer

Three columns, then a bottom line.

- Column 1: icon + “SlicerDDSurfer”, “Diffusion MRI workflows in 3D Slicer.” [F1 README:5], “© 2026 Chengjin Li and the
  SlicerDDSurfer contributors.” [F8 LICENSE:1-3]
- Column 2 “Licence”: “Project code: BSD-3-Clause. Bundled components keep their own licences and notices. Model
  weights, templates and subject-derived data are not relicensed by the project licence.” Links “LICENSE.txt”,
  “Third-party notices” → repo `blob/main/LICENSE.txt`, `blob/main/THIRD_PARTY_NOTICES.md`. [F8]
- Column 3 “About the film” [F9 credits]: “Data: EDDEN, OpenNeuro ds004666 (CC0), Manzano-Patron et al., Imaging
  Neuroscience 2024.” (links: “ds004666” → `https://openneuro.org/datasets/ds004666`, “2024” paper →
  `https://doi.org/10.1162/imag_a_00060`). “Run: SlicerDDSurfer a48fc95 from the command line, with 3D Slicer 5.12.4,
  SlicerDMRI and PyTorch on macOS (x86_64 under Rosetta, CPU only). This run is outside the Linux validation and needed
  local run-time workarounds that are not in the repository. Bundles and screenshot: runs of 23 October 2026 on the
  same Mac, from a repository checkout.” “Made with lemo-opuscar (MIT) and
  three.js (MIT). Voice: Kokoro TTS (Apache-2.0). Fonts: Inter, Inter Tight, IBM Plex Mono (SIL OFL 1.1). Music and
  sound: original, synthesized in code.” (link “lemo-opuscar” → `https://github.com/lemomo-ai/lemo-opuscar`)
  [F5 MAKING_OF:73-74, 144-145, 168; P/runtime/STATUS.md]
- Bottom line (small): “Static site: no cookies, no trackers, no third-party requests.” and “Back to top” `#top`.

## 6. Allowed external URLs (exact set)

`https://github.com/ChengjinLii/SlicerDDSurfer`, `https://github.com/ChengjinLii/SlicerDDSurfer.git` (in the code
block only), `https://github.com/ChengjinLii/SlicerDDSurfer/blob/main/` + one of `CITATION.cff`, `LICENSE.txt`,
`THIRD_PARTY_NOTICES.md`, `docs/setup/QUICKSTART.md`, `docs/setup/SUPPORT_MATRIX.md`,
`docs/assets/MODEL_ASSET_TERMS.md`, `docs/architecture/IMPLEMENTATION_PROVENANCE.md`;
the DOI links of the reference list (§5.10) and the four ISMRM archive pages (`https://archive.ismrm.org/<year>/<abstract>.html`),
`https://doi.org/10.1002/advs.76596`, `https://doi.org/10.1016/j.neuroimage.2021.117934`,
`https://doi.org/10.1109/TMI.2023.3331691`, `https://doi.org/10.1162/imag_a_00060`,
`https://openneuro.org/datasets/ds004666`, `https://github.com/lemomo-ai/lemo-opuscar`,
`https://chengjinlii.github.io/SlicerDDSurfer/` and `https://chengjinlii.github.io/SlicerDDSurfer/assets/img/social.jpg`
(head meta and `<link rel="canonical">` only; `404.html` may also link home with the first one, because GitHub Pages
serves it at nested paths where relative links break, so `404.html` inlines its CSS). Nothing is loaded from any external host.

## 7. Tests (must pass before push)

### 7.1 `Testing/Website/check_site.py` (run from the repository root: `python3 -I Testing/Website/check_site.py [--film PATH]`; exit 1 on any FAIL)

Prints one `PASS name` / `FAIL name - reason` line per check; `SKIP` only where stated.

1. `html_basics`: `docs/site/index.html` and `docs/site/404.html` parse; doctype, `lang="en"`, `<title>`, viewport,
   description; exactly one `<h1>` in index; heading levels never skip (h2 → h4); every `id` is unique.
2. `local_refs`: every local `href`, `src`, `poster`, `srcset` and CSS `url()` resolves to a file inside `docs/site/`;
   every `#fragment` link resolves to an `id` in the page.
3. `external_urls`: every absolute URL in `docs/site/` (HTML, CSS, JS) is in the §6 set; `<link>`, `<script>`, `<img>`,
   `<source>`, `<video>` and CSS `url()`/`@import` never point to another host, except `<link rel="canonical">` with
   the §6 Pages URL. Inline SVG carries no `xmlns` (or namespace URIs are exempt).
4. `film_bytes`: `docs/site/media/SlicerDSpace_promo_1080p.mp4` is 23,356,794 bytes and, when `--film` is given, equal in
   SHA-256 to that file.
5. `film_faststart`: top-level MP4 boxes: `ftyp` first, `moov` before `mdat` (pure Python box scan).
6. `film_probe`: with `ffprobe` on PATH: h264 1920×1080 yuv420p, aac, duration 64.8 ± 0.05 s. `SKIP` without ffprobe.
7. `transcript_matches_srt`: the transcript list equals the 13 cue texts of `docs/site/media/SlicerDSpace.en.srt`, in order.
8. `sizes`: no file in `docs/site/` is over 50 MB; `docs/site/` is under 100 MB in total.
9. `images`: every `<img>` has non-empty `alt`, integer `width` and `height`; each image file's pixel size matches its
   attributes' aspect ratio within 2% (PNG/JPEG header parse, stdlib only).
    In checks 10 and 11, “visible text” means the text content of both pages plus every `alt`, `aria-label`,
    `title` attribute, `<title>`, `<meta name="description">`/`og:*` content, and the string literals in `site.js`.
10. `claims_required`: the visible text contains each of: “at analysis time”, “Private pre-release”, “not a clinical
    assessment”, “not certified” (case-insensitive), “24,000 of 130,780”, “not installed silently”, “in preparation”, “display subset”, “not acceptance by Slicer”,
    “does not show them”, “No expert has reviewed”, “SNR is higher than a typical single scan”, “workarounds that are
    not in the repository”, “must share one voxel grid”.
11. `claims_banned`: the visible text matches none of (case-insensitive): `T1w?-free`, `T1-weighted-free`, `fully
    automatic`, `no registration`, `no preprocessing`, `U-shaped`, `\bthe first\b`, `\bthe only\b`, `clinical-grade`,
    `diagnos`, `FDA`, `available now`, `open[- ]source`, `\bfaster\b`, `\baccura(te|cy)\b`,
    `state[- ]of[- ]the[- ]art`, `cross-platform`, `(install|get|download)\w*\s+(SlicerDDSurfer|it)\b.{0,40}Extensions?
    (Manager|Catalog|Index)`, `(?<!not )(available )?(in|from|through) the (official )?Extensions? (Manager|Catalog)`
    (the allowed copy “not in the official Extensions Catalog” must pass). Also: `Extension Manager` occurs exactly
    once, inside “Install SlicerDMRI and PyTorch from Slicer’s Extension Manager.”
12. `no_local_paths`: no file in `docs/site/` contains `/Volumes/`, `/Users/`, `/private/tmp`, `file://`.
13. `links_safe`: every `target="_blank"` link has `rel` containing `noopener`.
14. `fonts_licensed`: every `.woff2` in `docs/site/assets/fonts/` has its OFL text next to it.

### 7.2 `Testing/Website/browser_check.mjs` (run: `npm ci --prefix Testing/Website`, then `node Testing/Website/browser_check.mjs`; `PLAYWRIGHT_CORE=<dir of playwright-core>` is optional)

Serves `docs/site/` from a local HTTP server with Range support (to match GitHub Pages), opens it in Google Chrome
(`channel: 'chrome'`; Playwright's Chromium lacks H.264/AAC), and checks at 1440×900 and 390×844 (and 320×640; and 861, 900,
1023, 1024 and 1100 px wide for header fit and no overlap of brand, nav links, chip and Menu button):

1. no console error; no failed request, except `net::ERR_ABORTED` on `media/*.mp4` (Chrome cancels and re-issues
   byte-range requests for video); no response with status ≥ 400; no request to another host;
2. `document.documentElement.scrollWidth <= innerWidth`;
3. every `<img>` is `complete` with `naturalWidth > 0` after scrolling the page through;
4. the video: `readyState >= 1`, `duration` 64.8 ± 0.05 s, `videoWidth` 1920; `canPlayType` for
   `video/mp4; codecs="avc1.640032, mp4a.40.2"` (High, level 5.0, as ffprobe reports) is not empty;
5. clicking the play button starts playback (`currentTime` advances within 3 s; muted in the test only) and shows
   `controls`;
6. at 390px the Menu button opens and closes the nav with the keyboard;
7. full-page screenshots to `Testing/Website/out/` (git-ignored) for a human look;
8. at 1440 and 390 px the reference list [4] to [35] is in a closed `<details>`, and a click on the `[21]` link in the
   integration table opens it and shows `#ref-21` in the upper half of the viewport once scrolling stops; without JS,
   loading `index.html#ref-21` opens it.

## 8. `docs/website/README.md` must say

1. What the folder is: the source notes of the project site, which is served from `docs/site/` on `main`; the site was
   made from the repo docs at a48fc95 and the delivered promo film (render 8).
2. How to publish (owner only): Settings → Pages → Source: GitHub Actions, then run the `Website` workflow manually; it
   publishes only `docs/site/` from `main`. The site is public
   once on, even while the repository is private; GitHub Pages on a private repository needs a paid plan.
3. How to test: the two commands in §7.
4. “Revisit at public release”: the status chip, the hero status line, the Status section, the “private” callout in
   Get started, the og:url/og:image/canonical URLs if the address differs, the Slicer version lines (README says 5.10,
   the film used 5.12.4; `--slicer-version` takes the form `Slicer-5.10` and the module path follows it).
   Before publishing: nobody has listened to the film's final mix yet (the ASR check is automatic); listen to it first.
   This is an owner note, not page copy.
6. Links into the repository use `blob/main`, so they follow `main`, not a48fc95.
5. Image crop boxes as finally used (§3).

## 9. Changes in v2 (Opus spec review, round 1)

Blocking items fixed: B1 unique ids (`film-video`; test); B2 slicer.jpg at 56.8 s, before the fade; B3 Outputs H2 and the
AVG_mag SNR caveat; B4 the film-run line names the CLI start and the local workarounds; B5 “Install dependencies …”
(an instruction, not an automatic install); B6 Extension Index JSON is a draft, not a submitted entry; B7 mask hedge
and “must share”; B8 Extension Manager / Catalog test; B9 ERR_ABORTED exemption for video range requests.
Non-blocking items taken in the same edit (copy precision or test scope): N1, N3, N4 (overflow widths), N5, N6, N7,
N8, N9, N10, N11, N13, N14, N17. Not taken (recorded in `README.md` → “Review backlog”): N2 (rule text only; boxes
fine), N12 (surfaces frame at 35.4 s), N15 (tool names), N16 (no inline SVG with xmlns is used), N18, N19.

Code review round 1 (commit 935a5d9): Opus B1 removed the “model-asset repository … private” clause from §5.9 (that
repository is public); Codex B1 moved the nav fold from ≤ 860px to ≤ 1023px and added the header overlap check to
§7.2. Non-blocking items are in `README.md` → “Review backlog”.


Editable artwork, source renders and generation scripts are maintained in the independent [paper-latex repository](https://github.com/ChengjinLii/paper-latex). This software checkout retains reviewed website images and their SHA256 receipts.
