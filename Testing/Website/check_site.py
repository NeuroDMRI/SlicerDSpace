#!/usr/bin/env python3
"""Static checks for the SlicerDSpace project site (SPEC section 7.1).

Run from the repository root: python3 -I Testing/Website/check_site.py [--film PATH]
Python 3 standard library only. ffprobe is optional.
Prints one PASS / FAIL / SKIP line per check. Exit code 1 on any FAIL.
"""
import hashlib
import html
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOCS = os.path.join(ROOT, "docs", "site")
INDEX = os.path.join(DOCS, "index.html")
PAGE404 = os.path.join(DOCS, "404.html")
CHINESE = os.path.join(DOCS, "zh.html")
FILM = os.path.join(DOCS, "media", "SlicerDSpace_promo_1080p.mp4")
SRT = os.path.join(DOCS, "media", "SlicerDSpace.en.srt")
FILM_BYTES = 24799408

REPO = "https://github.com/NeuroDMRI/SlicerDSpace"
BLOB = REPO + "/blob/main/"
ALLOWED_URLS = {
    REPO,
    REPO + ".git",
    BLOB + "CITATION.cff",
    BLOB + "LICENSE.txt",
    BLOB + "THIRD_PARTY_NOTICES.md",
    BLOB + "docs/setup/QUICKSTART.md",
    BLOB + ".agents/skills/slicerdspace/SKILL.md",
    BLOB + ".agents/skills/slicerdspace/references/usage.md",
    BLOB + "docs/setup/CLI_TEMPLATES.md",
    BLOB + "docs/setup/SUPPORT_MATRIX.md",
    BLOB + "docs/assets/MODEL_ASSET_TERMS.md",
    BLOB + "docs/architecture/IMPLEMENTATION_PROVENANCE.md",
    BLOB + "docs/release/PERFORMANCE_OPTIMIZATIONS.md",
    "https://doi.org/10.1002/advs.76596",
    "https://doi.org/10.1016/j.neuroimage.2021.117934",
    "https://doi.org/10.1109/TMI.2023.3331691",
    "https://doi.org/10.1162/imag_a_00060",
    "https://openneuro.org/datasets/ds004666",
    "https://github.com/lemomo-ai/lemo-opuscar",
    # References section (task: consolidated reference list). DOIs checked with curl on 2026-10-23.
    "https://archive.ismrm.org/2010/1670.html",
    "https://archive.ismrm.org/2017/3541.html",
    "https://archive.ismrm.org/2019/0555.html",
    "https://archive.ismrm.org/2021/2472.html",
    "https://doi.org/10.1002/mrm.20508",
    "https://doi.org/10.1002/mrm.22055",
    "https://doi.org/10.1007/978-3-031-43993-3_40",
    "https://doi.org/10.1007/978-3-642-33454-2_16",
    "https://doi.org/10.1007/s00429-015-1179-4",
    "https://doi.org/10.1016/j.mri.2012.05.001",
    "https://doi.org/10.1016/j.neuroimage.2004.07.037",
    "https://doi.org/10.1016/j.neuroimage.2012.01.021",
    "https://doi.org/10.1016/j.neuroimage.2012.03.072",
    "https://doi.org/10.1016/j.neuroimage.2012.06.005",
    "https://doi.org/10.1016/j.neuroimage.2014.07.061",
    "https://doi.org/10.1016/j.neuroimage.2014.09.053",
    "https://doi.org/10.1016/j.neuroimage.2014.10.026",
    "https://doi.org/10.1016/j.neuroimage.2015.06.092",
    "https://doi.org/10.1016/j.neuroimage.2018.06.027",
    "https://doi.org/10.1016/j.neuroimage.2019.04.015",
    "https://doi.org/10.1016/j.neuroimage.2019.116137",
    "https://doi.org/10.1016/j.neuroimage.2022.119423",
    "https://doi.org/10.1016/j.neuroimage.2022.119474",
    "https://doi.org/10.1101/108795",
    "https://doi.org/10.1109/TMI.2007.906785",
    "https://doi.org/10.1109/TMI.2010.2048121",
    "https://doi.org/10.1200/CCI.19.00141",
    "https://doi.org/10.18112/openneuro.ds004666.v1.0.8",
    "https://doi.org/10.3389/fninf.2014.00008",
    "https://doi.org/10.5281/zenodo.2648292",
    "https://neurodmri.github.io/SlicerDSpace/",
    "https://neurodmri.github.io/SlicerDSpace/zh.html",
    "https://neurodmri.github.io/SlicerDSpace/assets/img/social.jpg",
}

# The reviewed bibliography is the allowlist for scientific/software citations.
# It is maintained outside the public site; arbitrary new HTML links still fail.
with open(os.path.join(ROOT, "docs", "website", "references.json"), encoding="utf-8") as catalog:
    REFERENCE_CATALOG = json.load(catalog)
ALLOWED_URLS.update(row["url"] for row in REFERENCE_CATALOG["references"])

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
        "meta", "source", "track", "wbr"}
SKIP_TEXT = {"script", "style", "head", "title", "noscript"}


# ---------------------------------------------------------------- parsing
class Page(HTMLParser):
    """Collects what the checks need from one HTML file."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tags = []          # (tag, attrs dict, line)
        self.stack = []
        self.errors = []
        self.ids = set()
        self.headings = []      # (level, text)
        self.text = []          # visible text pieces
        self.style_blocks = []
        self.script_blocks = []
        self._skip = 0
        self._in_style = False
        self._in_script = False
        self._heading = None
        self.transcript = []    # li texts inside details#transcript
        self._in_transcript = False
        self._transcript_depth = 0
        self._li = None
        self.doctype = ""
        self.extra_text = []    # alt, aria-label, title attributes, meta content
        self.title_text = []
        self._in_title = False

    def handle_decl(self, decl):
        self.doctype = decl

    def handle_starttag(self, tag, attrs):
        a = {k: (v if v is not None else "") for k, v in attrs}
        self.tags.append((tag, a, self.getpos()[0]))
        for attr in ("alt", "aria-label", "title"):
            if a.get(attr):
                self.extra_text.append(a[attr])
        if tag == "meta" and a.get("content"):
            key = a.get("name", "") or a.get("property", "")
            if key == "description" or key.startswith(("og:", "twitter:")):
                self.extra_text.append(a["content"])
        if tag == "title":
            self._in_title = True
        if "id" in a:
            if a["id"] in self.ids:
                self.errors.append("duplicate id %r" % a["id"])
            self.ids.add(a["id"])
        if tag not in VOID:
            self.stack.append(tag)
        if tag in SKIP_TEXT:
            self._skip += 1
        if tag == "style":
            self._in_style = True
            self.style_blocks.append("")
        if tag == "script":
            self._in_script = True
            self.script_blocks.append("")
        if re.fullmatch(r"h[1-6]", tag):
            self._heading = (int(tag[1]), [])
        if tag == "details" and a.get("id") == "transcript":
            self._in_transcript = True
            self._transcript_depth = len(self.stack)
        if self._in_transcript and tag == "li":
            self._li = []

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if tag in self.stack:
            while self.stack and self.stack[-1] != tag:
                self.errors.append("unclosed <%s> before </%s> (line %d)" % (
                    self.stack.pop(), tag, self.getpos()[0]))
            self.stack.pop()
        else:
            self.errors.append("stray </%s> (line %d)" % (tag, self.getpos()[0]))
        if tag in SKIP_TEXT:
            self._skip -= 1
        if tag == "title":
            self._in_title = False
        if tag == "style":
            self._in_style = False
        if tag == "script":
            self._in_script = False
        if self._heading and tag == "h%d" % self._heading[0]:
            self.headings.append((self._heading[0], "".join(self._heading[1]).strip()))
            self._heading = None
        if self._in_transcript and tag == "li" and self._li is not None:
            self.transcript.append(" ".join("".join(self._li).split()))
            self._li = None
        if self._in_transcript and tag == "details":
            self._in_transcript = False

    def handle_data(self, data):
        if self._in_title:
            self.title_text.append(data)
        if self._in_style:
            self.style_blocks[-1] += data
        if self._in_script:
            self.script_blocks[-1] += data
        if self._heading is not None:
            self._heading[1].append(data)
        if self._li is not None:
            self._li.append(data)
        if not self._skip:
            self.text.append(data)

    def close(self):
        super().close()
        for t in self.stack:
            self.errors.append("unclosed <%s> at end of file" % t)

    def visible_text(self):
        return " ".join(" ".join(self.text).split())

    def wide_text(self):
        """Text content plus alt, aria-label, title attributes, <title> and meta content."""
        parts = [self.visible_text()] + self.extra_text + ["".join(self.title_text)]
        return " ".join(" ".join(parts).split())


def read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


JS_FILE = os.path.join(DOCS, "assets", "js", "site.js")
JS_STRING = re.compile(r"'((?:[^'\\\n]|\\.)*)'|\"((?:[^\"\\\n]|\\.)*)\"")


def js_literals():
    """String literals in site.js (comments removed first)."""
    if not os.path.isfile(JS_FILE):
        return []
    src = read(JS_FILE)
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    src = re.sub(r"(?m)^\s*//.*$", " ", src)
    return [m.group(1) if m.group(1) is not None else m.group(2) for m in JS_STRING.finditer(src)]


def wide_text(pages):
    """The 'visible text' of SPEC 7.1 checks 10 and 11."""
    parts = [p.wide_text() for _, p in pages] + js_literals()
    return " ".join(" ".join(parts).split())


def parse(path):
    p = Page()
    raw = read(path)
    p.feed(raw)
    p.close()
    return p, raw


def all_files(top=DOCS):
    for d, _, names in os.walk(top):
        for n in names:
            yield os.path.join(d, n)


# ---------------------------------------------------------------- results
results = []


def check(name):
    def deco(fn):
        def run(*a, **k):
            try:
                out = fn(*a, **k)
            except Exception as exc:  # a crashing check is a failing check
                results.append(("FAIL", name, "exception: %r" % (exc,)))
                return
            if out is None:
                results.append(("PASS", name, ""))
            elif isinstance(out, tuple) and out[0] == "SKIP":
                results.append(("SKIP", name, out[1]))
            else:
                results.append(("FAIL", name, out))
        run.__name__ = fn.__name__
        return run
    return deco


def fail_list(items):
    return "; ".join(items[:8]) + (" (+%d more)" % (len(items) - 8) if len(items) > 8 else "")


# ---------------------------------------------------------------- checks
@check("html_basics")
def html_basics(idx, raw_idx, p404, raw_404, *localized):
    problems = []
    entries = [("index", idx, raw_idx, "en"), ("404", p404, raw_404, "en"), *localized]
    for label, p, raw, language in entries:
        dup = [e for e in p.errors if e.startswith("duplicate id")]
        if dup:
            problems.append("%s: every id must be unique: %s" % (label, fail_list(dup)))
        other = [e for e in p.errors if e not in dup]
        if other:
            problems.append("%s: %s" % (label, fail_list(other)))
        if not raw.lstrip().lower().startswith("<!doctype html>"):
            problems.append("%s: missing <!doctype html>" % label)
        html_tags = [a for t, a, _ in p.tags if t == "html"]
        if not html_tags or html_tags[0].get("lang") != language:
            problems.append('%s: <html lang="%s"> missing' % (label, language))
        titles = re.findall(r"<title>(.*?)</title>", raw, re.S)
        if not titles or not titles[0].strip():
            problems.append("%s: empty <title>" % label)
        metas = {a.get("name"): a for t, a, _ in p.tags if t == "meta" and a.get("name")}
        if "viewport" not in metas:
            problems.append("%s: no viewport meta" % label)
        if not metas.get("description", {}).get("content", "").strip():
            problems.append("%s: no description meta" % label)
        if not any(t == "meta" and a.get("charset", "").lower() == "utf-8" for t, a, _ in p.tags):
            problems.append("%s: no charset utf-8" % label)
    h1 = [h for h in idx.headings if h[0] == 1]
    if len(h1) != 1:
        problems.append("index: %d <h1>, need exactly 1" % len(h1))
    for label, p, _, _ in entries:
        prev = 0
        for level, text in p.headings:
            if level > prev + 1:
                problems.append("%s: heading level jumps h%d -> h%d (%r)" % (label, prev, level, text[:40]))
            prev = level
    return fail_list(problems) if problems else None


CSS_URL = re.compile(r"url\(\s*(['\"]?)(.*?)\1\s*\)", re.I)
CSS_IMPORT = re.compile(r"@import\s+(?:url\()?\s*['\"]?([^'\")\s;]+)", re.I)


def is_absolute(u):
    return bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", u)) or u.startswith("//")


def resolve_local(base_file, ref):
    """Return the absolute file path a local reference points to."""
    path = unquote(urlsplit(ref).path)
    if path == "":
        return base_file
    if path.startswith("/"):
        target = os.path.join(DOCS, path.lstrip("/"))
    else:
        target = os.path.normpath(os.path.join(os.path.dirname(base_file), path))
    if os.path.isdir(target):
        target = os.path.join(target, "index.html")
    return target


@check("local_refs")
def local_refs(pages):
    problems = []
    for path, p in pages:
        rel = os.path.relpath(path, ROOT)
        refs = []
        for tag, a, line in p.tags:
            for attr in ("href", "src", "poster"):
                if attr in a:
                    refs.append((attr, a[attr], line))
            if "srcset" in a:
                for part in a["srcset"].split(","):
                    if part.strip():
                        refs.append(("srcset", part.strip().split()[0], line))
        for block in p.style_blocks:
            for m in CSS_URL.finditer(block):
                refs.append(("css-url", m.group(2), 0))
        for attr, ref, line in refs:
            if not ref or is_absolute(ref):
                continue
            if ref.startswith("#"):
                if ref != "#" and ref[1:] not in p.ids:
                    problems.append("%s:%s %s=%r has no matching id" % (rel, line, attr, ref))
                continue
            target = resolve_local(path, ref)
            inside = os.path.commonpath([DOCS, target]) == DOCS
            if not inside or not os.path.isfile(target):
                problems.append("%s:%s %s=%r does not resolve to a file in docs/" % (rel, line, attr, ref))
    for css in (f for f in all_files() if f.endswith(".css")):
        text = read(css)
        for m in CSS_URL.finditer(text):
            ref = m.group(2)
            if not ref or is_absolute(ref) or ref.startswith("#"):
                continue
            target = resolve_local(css, ref)
            if not os.path.isfile(target):
                problems.append("%s url(%r) missing" % (os.path.relpath(css, ROOT), ref))
    return fail_list(problems) if problems else None


# XML namespace URIs (for example in inline SVG) are identifiers, not requests.
NAMESPACE_PREFIXES = ("http://www.w3.org/",)

URL_RE = re.compile(r"https?://[^\s\"'<>`\\]+")


@check("external_urls")
def external_urls(pages):
    problems = []
    texts = []
    for path, p in pages:
        texts.append((path, read(path)))
    for f in all_files():
        if f.endswith((".css", ".js")):
            texts.append((f, read(f)))
    for path, text in texts:
        rel = os.path.relpath(path, ROOT)
        for m in URL_RE.finditer(text):
            url = m.group(0).rstrip(".,;:")
            # Parentheses are legal DOI characters; strip only an unmatched
            # closing delimiter, as in an unquoted CSS url(...).
            while url.endswith(")") and url.count(")") > url.count("("):
                url = url[:-1]
            if url.startswith(NAMESPACE_PREFIXES):
                continue
            if url not in ALLOWED_URLS:
                problems.append("%s: URL not in the allowed set: %s" % (rel, url))
        for m in re.finditer(r"(?<![:\w])//[a-zA-Z0-9.-]+\.[a-z]{2,}/", text):
            problems.append("%s: protocol-relative URL %s" % (rel, m.group(0)))
    # Resource-loading tags and CSS must stay on this host.
    for path, p in pages:
        rel = os.path.relpath(path, ROOT)
        for tag, a, line in p.tags:
            if tag in ("link", "script", "img", "source", "video"):
                for attr in ("href", "src", "poster", "srcset"):
                    v = a.get(attr, "")
                    if v and is_absolute(v):
                        if v.startswith("data:"):
                            continue
                        if tag == "link" and a.get("rel") in ("canonical", "alternate") and v in ALLOWED_URLS:
                            continue
                        problems.append("%s:%s <%s %s=%r> points to another host" % (rel, line, tag, attr, v))
    for f in all_files():
        if f.endswith(".css"):
            text = read(f)
            for m in CSS_URL.finditer(text):
                if is_absolute(m.group(2)) and not m.group(2).startswith("data:"):
                    problems.append("%s: CSS url(%s) is absolute" % (os.path.relpath(f, ROOT), m.group(2)))
                if m.group(2).startswith("data:"):
                    pass
            for m in CSS_IMPORT.finditer(text):
                if is_absolute(m.group(1)):
                    problems.append("%s: CSS @import %s is absolute" % (os.path.relpath(f, ROOT), m.group(1)))
    return fail_list(sorted(set(problems))) if problems else None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@check("film_bytes")
def film_bytes(film_arg):
    if not os.path.isfile(FILM):
        return "film file missing"
    size = os.path.getsize(FILM)
    if size != FILM_BYTES:
        return "size is %d bytes, need %d" % (size, FILM_BYTES)
    if film_arg:
        if not os.path.isfile(film_arg):
            return "--film path not found: %s" % film_arg
        if sha256(FILM) != sha256(film_arg):
            return "SHA-256 differs from %s" % film_arg
    return None


@check("film_faststart")
def film_faststart():
    boxes = []
    with open(FILM, "rb") as f:
        size_total = os.fstat(f.fileno()).st_size
        pos = 0
        while pos < size_total:
            f.seek(pos)
            head = f.read(16)
            if len(head) < 8:
                return "truncated box header at %d" % pos
            size, kind = struct.unpack(">I4s", head[:8])
            kind = kind.decode("latin-1")
            if size == 1:
                size = struct.unpack(">Q", head[8:16])[0]
            elif size == 0:
                size = size_total - pos
            if size < 8:
                return "bad box size %d for %r at %d" % (size, kind, pos)
            boxes.append(kind)
            pos += size
    if not boxes or boxes[0] != "ftyp":
        return "first box is %r, need ftyp (boxes: %s)" % (boxes[:1], boxes)
    if "moov" not in boxes or "mdat" not in boxes:
        return "moov or mdat missing (boxes: %s)" % boxes
    if boxes.index("moov") > boxes.index("mdat"):
        return "moov is after mdat (boxes: %s)" % boxes
    return None


@check("film_probe")
def film_probe():
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return ("SKIP", "ffprobe not on PATH")
    out = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries",
         "format=duration:stream=codec_type,codec_name,width,height,pix_fmt",
         "-of", "json", FILM],
        capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    streams = info.get("streams", [])
    video = [s for s in streams if s.get("codec_type") == "video"]
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    problems = []
    if not video:
        problems.append("no video stream")
    else:
        v = video[0]
        if (v.get("codec_name"), v.get("width"), v.get("height"), v.get("pix_fmt")) != (
                "h264", 1920, 1080, "yuv420p"):
            problems.append("video is %s %sx%s %s" % (
                v.get("codec_name"), v.get("width"), v.get("height"), v.get("pix_fmt")))
    if not audio or audio[0].get("codec_name") != "aac":
        problems.append("audio is not aac")
    dur = float(info["format"]["duration"])
    if abs(dur - 72.0) > 0.05:
        problems.append("duration %.3f s, need 72.0 +/- 0.05" % dur)
    return "; ".join(problems) if problems else None


def srt_cues(path):
    blocks = re.split(r"\n\s*\n", read(path).replace("\r\n", "\n").strip())
    cues = []
    for b in blocks:
        lines = b.split("\n")
        if len(lines) >= 3 and "-->" in lines[1]:
            cues.append(" ".join(l.strip() for l in lines[2:]))
    return cues


@check("minimal_film_controls")
def minimal_film_controls(idx):
    videos = [attributes for tag, attributes, _ in idx.tags if tag == 'video']
    if len(videos) != 1 or videos[0].get('preload') != 'none':
        return 'The film must load only on request (preload=none)'
    if idx.transcript or any(tag == 'details' and attributes.get('id') == 'transcript'
                             for tag, attributes, _ in idx.tags):
        return 'The film transcript was removed from the homepage'
    if any(tag == 'a' and attributes.get('href', '').endswith('.srt') for tag, attributes, _ in idx.tags):
        return 'The homepage must not offer subtitle downloads'
    return None


@check("sizes")
def sizes():
    total = 0
    for f in all_files():
        s = os.path.getsize(f)
        total += s
        if s > 50 * 1024 * 1024:
            return "%s is %d bytes (limit 50 MB)" % (os.path.relpath(f, ROOT), s)
    if total >= 100 * 1024 * 1024:
        return "docs/ is %d bytes (limit 100 MB)" % total
    return None


def image_size(path, is_bytes=False):
    if is_bytes:
        data = path
    else:
        with open(path, "rb") as f:
            data = f.read()
    if b'<svg' in data[:512]:
        root = ET.fromstring(data)
        viewbox = [float(value) for value in root.attrib['viewBox'].replace(',', ' ').split()]
        if len(viewbox) != 4 or viewbox[2] <= 0 or viewbox[3] <= 0:
            raise ValueError('SVG needs a positive viewBox')
        return viewbox[2], viewbox[3]
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        w, h = struct.unpack(">II", data[16:24])
        return w, h
    if data[:2] == b"\xff\xd8":
        i = 2
        while i < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            seglen = struct.unpack(">H", data[i + 2:i + 4])[0]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                h, w = struct.unpack(">HH", data[i + 5:i + 9])
                return w, h
            i += 2 + seglen
    raise ValueError("not a PNG, JPEG or SVG")


@check("images")
def images(pages):
    problems = []
    for path, p in pages:
        rel = os.path.relpath(path, ROOT)
        for tag, a, line in p.tags:
            if tag != "img":
                continue
            src = a.get("src", "")
            if not a.get("alt", "").strip():
                problems.append("%s:%s %s has no alt" % (rel, line, src))
            try:
                w, h = int(a["width"]), int(a["height"])
            except (KeyError, ValueError):
                problems.append("%s:%s %s width/height missing or not integers" % (rel, line, src))
                continue
            # Above-the-fold branding loads eagerly; gallery images remain lazy.
            eager = {"assets/img/logo-ddsurfer.png", "assets/img/slicerdspace-wordmark-light.svg"}
            if src not in eager and a.get("loading") != "lazy":
                problems.append("%s:%s %s lacks loading=lazy" % (rel, line, src))
            if src.startswith("data:"):
                import base64
                raw = base64.b64decode(src.split(",", 1)[1])
                iw, ih = image_size(raw, is_bytes=True)
            else:
                target = resolve_local(path, src)
                if not os.path.isfile(target):
                    problems.append("%s:%s %s missing" % (rel, line, src))
                    continue
                iw, ih = image_size(target)
            if abs((iw / ih) / (w / h) - 1) > 0.02:
                problems.append("%s:%s %s file %dx%d vs attributes %dx%d" % (rel, line, src, iw, ih, w, h))
    return fail_list(problems) if problems else None


# Keep substantive installation and display scopes; removed Status prose is not
# a homepage requirement. Per-run scientific evidence remains in source records.
REQUIRED = ["Open full-size SVG",
            "not installed silently",
            "anatomical review", "must share one voxel grid"]
BANNED = [r"T1w?-free", r"T1-weighted-free", r"fully automatic", r"no registration",
          r"no preprocessing", r"U-shaped", r"\bthe first\b", r"\bthe only\b",
          r"clinical-grade", r"diagnos", r"FDA", r"available now", r"open[- ]source",
          r"\bfaster\b", r"\baccura(te|cy)\b", r"state[- ]of[- ]the[- ]art", r"cross-platform",
          r"(install|get|download)\w*\s+(SlicerDSpace|it)\b.{0,40}Extensions? (Manager|Catalog|Index)",
          r"(?<!not )(available )?(in|from|through) the (official )?Extensions? (Manager|Catalog)",
          # Opus code review r1 B1: the SlicerDDSurfer-models repository is public
          # (R/docs/release/VALIDATION_20260816.md:36), so the page must not call it private.
          r"model[- ]asset\w* repositor(y|ies)\b[^.]{0,40}\bprivate"]
EXT_MANAGER_SENTENCE = "Install SlicerDMRI and PyTorch from Slicer\u2019s Extension Manager."


@check("claims_required")
def claims_required(pages):
    text = wide_text(pages).lower()
    missing = [r for r in REQUIRED if r.lower() not in text]
    return "missing phrases: %s" % missing if missing else None


@check("claims_banned")
def claims_banned(pages):
    problems = []
    text = wide_text(pages)
    # Paper titles (e.g. dcm2niix's "The first step...") are not product claims.
    for row in REFERENCE_CATALOG["references"]:
        text = text.replace(" ".join(row["citation"].split()), "")
    for pat in BANNED:
        for m in re.finditer(pat, text, re.I):
            ctx = text[max(0, m.start() - 30):m.end() + 30]
            problems.append("%r in ...%s..." % (pat, ctx))
    n = len(re.findall(r"Extension Manager", text, re.I))
    if n != 1:
        problems.append("'Extension Manager' occurs %d times, expected exactly 1" % n)
    elif EXT_MANAGER_SENTENCE not in text:
        problems.append("'Extension Manager' is not inside %r" % EXT_MANAGER_SENTENCE)
    return fail_list(problems) if problems else None


@check("no_local_paths")
def no_local_paths():
    needles = [b"/Volumes/", b"/Users/", b"/private/tmp", b"file://"]
    problems = []
    for f in all_files():
        with open(f, "rb") as fh:
            data = fh.read()
        for n in needles:
            if n in data:
                problems.append("%s contains %s" % (os.path.relpath(f, ROOT), n.decode()))
    return fail_list(problems) if problems else None


@check("links_safe")
def links_safe(pages):
    problems = []
    for path, p in pages:
        for tag, a, line in p.tags:
            if a.get("target") == "_blank" and "noopener" not in a.get("rel", "").split():
                problems.append("%s:%s target=_blank without rel=noopener (%s)" % (
                    os.path.basename(path), line, a.get("href", "")))
    return fail_list(problems) if problems else None


@check("fonts_licensed")
def fonts_licensed():
    fdir = os.path.join(DOCS, "assets", "fonts")
    woffs = [n for n in os.listdir(fdir) if n.endswith(".woff2")]
    if not woffs:
        return "no .woff2 files found"
    problems = []
    for n in woffs:
        family = n[:-len(".woff2")].split("-")[0]
        if not os.path.isfile(os.path.join(fdir, "OFL-%s.txt" % family)):
            problems.append("%s has no OFL-%s.txt" % (n, family))
    return fail_list(problems) if problems else None


@check("localization")
def localization():
    result = subprocess.run([sys.executable, '-I', os.path.join(ROOT, 'scripts/build_website_locales.py'),
                             '--check'], capture_output=True, text=True)
    if result.returncode:
        return result.stdout + result.stderr


# ---------------------------------------------------------------- main
def main(argv):
    film_arg = None
    if "--film" in argv:
        i = argv.index("--film")
        if i + 1 >= len(argv):
            print("usage: check_site.py [--film PATH]")
            return 2
        film_arg = argv[i + 1]
    idx, raw_idx = parse(INDEX)
    p404, raw_404 = parse(PAGE404)
    chinese, raw_chinese = parse(CHINESE)
    pages = [(INDEX, idx), (PAGE404, p404), (CHINESE, chinese)]

    html_basics(idx, raw_idx, p404, raw_404, ('zh', chinese, raw_chinese, 'zh-CN'))
    localization()
    local_refs(pages)
    external_urls(pages)
    film_bytes(film_arg)
    film_faststart()
    film_probe()
    minimal_film_controls(idx)
    sizes()
    images(pages)
    claims_required(pages)
    claims_banned(pages)
    no_local_paths()
    links_safe(pages)
    fonts_licensed()

    failed = 0
    for status, name, reason in results:
        if status == "PASS":
            print("PASS %s" % name)
        elif status == "SKIP":
            print("SKIP %s - %s" % (name, reason))
        else:
            failed += 1
            print("FAIL %s - %s" % (name, reason))
    print("%d checks, %d failed" % (len(results), failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
