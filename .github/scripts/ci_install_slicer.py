#!/usr/bin/env python3
"""Download and install a stock 3D Slicer release on a GitHub-hosted runner (CI matrix helper).

Source of truth: https://download.slicer.org/find?os=<linux|win|macosx>&stability=release&version=<x.y.z>
It returns JSON with a relative download_url, the revision and a sha512. The revision must match
REVISIONS below (the same table as .github/scripts/repackage_release_assets.py: 5.10 -> 34045,
5.12 -> 34645); a mismatch fails the step, because the SlicerDSpace layout depends on it.

Linux : tar.gz is extracted with tar.      Windows: NSIS installer, silent (/S /D=<dir>).
macOS : dmg is attached with hdiutil and Slicer.app is copied (x86_64 build; runs under Rosetta).

Writes slicer_exe, python_slicer, slicer_home and slicer_revision to $GITHUB_OUTPUT and prints them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://download.slicer.org"
USER_AGENT = "slicerdspace-ci"
REVISIONS = {"5.10": "34045", "5.12": "34645"}  # keep equal to repackage_release_assets.py
API_OS = {"linux": "linux", "win32": "win", "darwin": "macosx"}


def log(message):
    print("[ci-install-slicer] " + message, flush=True)


def api_os_name(platform=None):
    platform = platform or sys.platform
    key = "linux" if platform.startswith("linux") else platform
    if key not in API_OS:
        raise SystemExit("Unsupported platform for the Slicer download API: %r" % platform)
    return API_OS[key]


def minor_key(version):
    parts = version.split(".")
    return ".".join(parts[:2])


def find_release(os_name, version, opener=None):
    """Query download.slicer.org and return the validated release record."""
    url = "%s/find?%s" % (API, urllib.parse.urlencode({"os": os_name, "stability": "release", "version": version}))
    opener = opener or (lambda u: urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": USER_AGENT}), timeout=60))
    with opener(url) as response:
        body = response.read().decode("utf-8", "replace")
    try:
        record = json.loads(body)
    except ValueError:
        raise SystemExit("download.slicer.org returned no JSON for os=%s version=%s (cause: unknown os name or version; "
                         "fix: use linux, win or macosx and an existing release). First bytes: %r" % (os_name, version, body[:120]))
    return validate_record(record, os_name, version)


def validate_record(record, os_name, version):
    for key in ("download_url", "revision", "sha512", "size", "version"):
        if not record.get(key):
            raise SystemExit("download.slicer.org record for %s %s has no %r: %s" % (os_name, version, key, record))
    if record["version"] != version:
        raise SystemExit("Asked for Slicer %s but the API returned %s" % (version, record["version"]))
    expected = REVISIONS.get(minor_key(version))
    if expected and str(record["revision"]) != expected:
        raise SystemExit("Slicer %s has revision %s, expected %s from the repackage table; update REVISIONS and the "
                         "installer remap only after checking the extension layout" % (version, record["revision"], expected))
    url = record["download_url"]
    if url.startswith("/"):
        url = API + url
    record = dict(record)
    record["download_url"] = url
    return record


def sha512_of(path):
    digest = hashlib.sha512()
    with open(str(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(record, dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)
    final = dest_dir / ("%s.pkg" % record["name"])
    if final.is_file() and final.stat().st_size == int(record["size"]) and sha512_of(final) == record["sha512"]:
        log("cache hit, sha512 verified: %s" % final)
        return final
    part = final.with_name(final.name + ".part")
    for attempt in range(1, 4):
        try:
            log("downloading %s (attempt %d/3, %.0f MB)" % (record["download_url"], attempt, int(record["size"]) / 1e6))
            request = urllib.request.Request(record["download_url"], headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=300) as response, open(str(part), "wb") as out:
                shutil.copyfileobj(response, out, 1 << 22)
            actual = sha512_of(part)
            if actual != record["sha512"]:
                raise RuntimeError("sha512 mismatch: %s != %s" % (actual, record["sha512"]))
            os.replace(str(part), str(final))
            return final
        except Exception as exc:  # noqa: BLE001 - retry any transport error
            log("download failed: %s" % exc)
            if part.exists():
                part.unlink()
            time.sleep(5 * attempt)
    raise SystemExit("Could not download a verified Slicer archive after 3 attempts")


def run(command, **kwargs):
    log("run: " + " ".join(str(c) for c in command))
    subprocess.run([str(c) for c in command], check=True, **kwargs)


def install_linux(archive, install_dir):
    install_dir.mkdir(parents=True, exist_ok=True)
    run(["tar", "-xzf", archive, "-C", install_dir])
    roots = [p for p in install_dir.iterdir() if p.is_dir() and (p / "Slicer").is_file()]
    if len(roots) != 1:
        raise SystemExit("Expected one Slicer-* folder with a Slicer launcher in %s, found %s" % (install_dir, roots))
    return roots[0], roots[0] / "Slicer", roots[0] / "bin" / "PythonSlicer"


def install_windows(archive, install_dir):
    if install_dir.exists():
        shutil.rmtree(str(install_dir))
    installer = archive.with_suffix(".exe")
    shutil.copyfile(str(archive), str(installer))
    # NSIS: /S silent; /D must be the last argument and unquoted. The path has no spaces (RUNNER_TEMP).
    command = [str(installer), "/S", "/D=" + str(install_dir)]
    log("run: " + " ".join(command))
    result = subprocess.run(command)
    log("installer exit code %s" % result.returncode)
    slicer = install_dir / "Slicer.exe"
    python = install_dir / "bin" / "PythonSlicer.exe"
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline and not (slicer.is_file() and python.is_file()):
        time.sleep(5)  # NSIS can hand over to a child process
    if not (slicer.is_file() and python.is_file()):
        raise SystemExit("Silent install did not produce %s and %s (installer exit code %s)" % (slicer, python, result.returncode))
    return install_dir, slicer, python


def install_macos(archive, install_dir):
    mount = install_dir.parent / "slicer-dmg-mount"
    mount.mkdir(parents=True, exist_ok=True)
    install_dir.mkdir(parents=True, exist_ok=True)
    dmg = archive.with_suffix(".dmg")
    shutil.copyfile(str(archive), str(dmg))
    # The Slicer dmg carries a license agreement; without "Y" on stdin hdiutil prints it and cancels the attach.
    run(["hdiutil", "attach", "-nobrowse", "-readonly", "-mountpoint", mount, dmg], input=b"Y\n")
    try:
        apps = sorted(mount.glob("Slicer*.app"))
        if not apps:
            raise SystemExit("No Slicer*.app inside the dmg: %s" % list(mount.iterdir()))
        target = install_dir / "Slicer.app"
        if target.exists():
            shutil.rmtree(str(target))
        run(["cp", "-R", apps[0], target])
    finally:
        subprocess.run(["hdiutil", "detach", str(mount), "-force"], check=False)
    contents = install_dir / "Slicer.app" / "Contents"
    return contents, contents / "MacOS" / "Slicer", contents / "bin" / "PythonSlicer"


def write_outputs(values):
    for key, value in values.items():
        log("%s=%s" % (key, value))
    target = os.environ.get("GITHUB_OUTPUT")
    if target:
        with open(target, "a", encoding="utf-8") as handle:
            for key, value in values.items():
                handle.write("%s=%s\n" % (key, value))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=True, help="Slicer release, for example 5.10.0")
    temp = Path(os.environ.get("RUNNER_TEMP") or os.environ.get("TMPDIR") or "/tmp")
    parser.add_argument("--download-dir", type=Path, default=temp / "slicer-dl", help="Cached archive folder (actions/cache path)")
    parser.add_argument("--install-dir", type=Path, default=temp / "slicer-install")
    args = parser.parse_args(argv)

    os_name = api_os_name()
    record = find_release(os_name, args.version)
    log("release: %s revision %s size %s" % (record["name"], record["revision"], record["size"]))
    archive = download(record, args.download_dir)
    installers = {"linux": install_linux, "win": install_windows, "macosx": install_macos}
    home, slicer_exe, python_slicer = installers[os_name](archive, args.install_dir)
    for path in (slicer_exe, python_slicer):
        if not path.is_file():
            raise SystemExit("Missing after install: %s" % path)
    write_outputs({"slicer_exe": slicer_exe, "python_slicer": python_slicer, "slicer_home": home,
                   "slicer_revision": record["revision"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
