#!/usr/bin/env python3
"""X8: record the highest GLIBC_ and GLIBCXX_ symbol versions that bundled ELF files require (Linux CI helper).

Scans every ELF file under the given roots with `objdump -T` (undefined dynamic symbols = what the file
needs from the host). Prints a table either way. Exit code 1 when any file needs GLIBC above the limit
(default 2.34). The limit is the documented Linux floor (Ubuntu 22.04+, Debian 12+, RHEL/Rocky/Alma 9+), set
from the scan of the released v0.1.1 asset: bundled mrcrop and the surface-runtime FreeSurfer tools need
2.34 and everything else needs 2.28 or less. Any bundle that needs more than 2.34 fails CI.
GLIBCXX is recorded only.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

GLIBC_RE = re.compile(r"\bGLIBC_(\d+(?:\.\d+)*)")
GLIBCXX_RE = re.compile(r"\bGLIBCXX_(\d+(?:\.\d+)*)")
DEFAULT_LIMIT = "2.34"


def version_tuple(text):
    return tuple(int(part) for part in text.split("."))


def max_version(versions):
    """Highest version as a string; integer tuple compare, so 2.4 < 2.28."""
    versions = list(versions)
    return max(versions, key=version_tuple) if versions else ""


def parse_objdump(text):
    """Return (max GLIBC, max GLIBCXX) of the UNDEFINED dynamic symbols in `objdump -T` output."""
    glibc, glibcxx = [], []
    for line in text.splitlines():
        if "*UND*" not in line:
            continue
        glibc += GLIBC_RE.findall(line)
        glibcxx += GLIBCXX_RE.findall(line)
    return max_version(glibc), max_version(glibcxx)


def is_elf(path):
    try:
        with open(str(path), "rb") as handle:
            return handle.read(4) == b"\x7fELF"
    except OSError:
        return False


def iter_elf_files(roots):
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and not path.is_symlink() and is_elf(path):
                yield path


def scan_file(path, objdump):
    result = subprocess.run([objdump, "-T", str(path)], capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        return {"file": str(path), "glibc": "", "glibcxx": "", "note": "objdump: " + (result.stderr.strip().splitlines() or ["failed"])[0]}
    glibc, glibcxx = parse_objdump(result.stdout)
    return {"file": str(path), "glibc": glibc, "glibcxx": glibcxx, "note": ""}


def violations(rows, limit):
    limit_tuple = version_tuple(limit)
    return [r for r in rows if r["glibc"] and version_tuple(r["glibc"]) > limit_tuple]


def format_table(rows, limit, top=40):
    ordered = sorted(rows, key=lambda r: (version_tuple(r["glibc"]) if r["glibc"] else (0,)), reverse=True)
    bad = violations(rows, limit)
    lines = ["max GLIBC required: %s    max GLIBCXX required: %s    files scanned: %d    above GLIBC %s: %d" % (
        max_version(r["glibc"] for r in rows if r["glibc"]) or "-",
        max_version(r["glibcxx"] for r in rows if r["glibcxx"]) or "-", len(rows), limit, len(bad)),
        "%-8s %-10s %s" % ("GLIBC", "GLIBCXX", "file")]
    shown = ordered[:top] + [r for r in bad if r not in ordered[:top]]
    for row in shown:
        lines.append("%-8s %-10s %s%s" % (row["glibc"] or "-", row["glibcxx"] or "-", row["file"], "   <-- above limit" if row in bad else ""))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="+", type=Path, help="Folders to scan, for example bundled-runtimes bundled-python")
    parser.add_argument("--limit", default=DEFAULT_LIMIT, help="Highest allowed GLIBC version (default 2.28)")
    parser.add_argument("--json", type=Path, help="Write all rows to this JSON file")
    parser.add_argument("--objdump", default=shutil.which("objdump") or "objdump")
    args = parser.parse_args(argv)

    files = sorted(iter_elf_files(args.roots))
    print("[glibc-scan] %d ELF files under %s" % (len(files), ", ".join(str(r) for r in args.roots)), flush=True)
    if not files:
        print("[glibc-scan] ERROR: no ELF files found; the roots are wrong or the install is empty", flush=True)
        return 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(lambda p: scan_file(p, args.objdump), files))
    print(format_table(rows, args.limit), flush=True)
    failed_to_read = [r for r in rows if r["note"]]
    if failed_to_read:
        print("[glibc-scan] %d file(s) objdump could not read (static or not dynamic); first: %s" % (
            len(failed_to_read), failed_to_read[0]["note"]), flush=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"limit": args.limit, "rows": rows}, indent=1) + "\n", encoding="utf-8")
    bad = violations(rows, args.limit)
    if bad:
        print("[glibc-scan] FAIL: %d file(s) need GLIBC above %s; the Linux bundle would not start on older hosts" % (len(bad), args.limit), flush=True)
        return 1
    print("[glibc-scan] PASS: no file needs GLIBC above %s" % args.limit, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
