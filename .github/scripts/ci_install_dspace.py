#!/usr/bin/env python3
"""Install SlicerDSpace into a stock Slicer the way users get it, then overlay this checkout (CI helper).

1. Download the released base asset for this OS (gh release download, SHA-256 checked against the release's
   SHA256SUMS.txt and against docs/website/installer_artifacts.json).
2. Run the SAME installer the website generates (shared.installer_script_builder.generate_installer_script)
   inside the target PythonSlicer, as the generated Install-SlicerDSpace.* wrapper does. It uses the
   downloaded asset from --offline-dir, so nothing is downloaded twice. No WMA atlas pack is selected.
3. Copy the runtime files of this checkout (scripts/extension_runtime/runtime_resources.json) over the
   installed module directory, one file at a time (the installed tree is never emptied), and prove that the
   installed code tree now has the same SHA-256 fingerprint as the checkout.

With --reinstall the script runs the SAME website installer a second time over the copy that the first run
installed (the asset from the first run must still be in --offline-dir: pass --keep-asset to the first run). The
reinstall fails when the installer fails, when it did not go over an installed copy, when SlicerDSpace is missing
afterwards, or when any Slicer settings file lists SlicerDSpace under ScheduledForUninstall. It then overlays this
checkout again, because the reinstall replaced the overlaid files with the released ones.

Prints and writes module_dir (the directory that holds SlicerDSpace.py) and ext_root to $GITHUB_OUTPUT.
Never run this against a Slicer you care about: it installs and overwrites files in that Slicer.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

RELEASE_TAG = "v0.1.1"
REPO = "NeuroDMRI/SlicerDSpace"
BASE_ASSETS = {
    "linux": ("linux", "SlicerDSpace-%s-Linux-x64.tar.gz", "linux_x64_base"),
    "win32": ("windows", "SlicerDSpace-%s-Windows-x64.zip", "windows_x64_base"),
    "darwin": ("macos", "SlicerDSpace-%s-macOS-arm64.tar.gz", "macos_arm64_base"),
}
EXEC_SNIPPET = (
    "import sys; bmk='#'+' __SLICERDSPACE_PY_BEGIN__'; emk='#'+' __SLICERDSPACE_PY_END__'; "
    "t=open(sys.argv[1], encoding='utf-8').read(); b=t.split(bmk,1)[1].split(emk,1)[0]; "
    "exec(compile(b, sys.argv[1], 'exec'), {'__name__': '__main__', '__file__': sys.argv[1]})"
)
# The website installer logs this line when it found an older SlicerDSpace and started Slicer a second time (finding B1).
REINSTALL_MARKER = "An older SlicerDSpace is installed."
INSTALLED_AT = re.compile(r"SlicerDSpace extension installed at: (.+)$", re.MULTILINE)


def log(message):
    print("[ci-install-dspace] " + message, flush=True)


def sha256_of(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def platform_key():
    return "linux" if sys.platform.startswith("linux") else sys.platform


def parse_sha256sums(text):
    table = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 64:
            table[parts[-1].lstrip("*")] = parts[0].lower()
    return table


def find_installed_module_dir(installer_output):
    """The installer prints 'SlicerDSpace extension installed at: <module dir>'."""
    match = INSTALLED_AT.findall(installer_output)
    return Path(match[-1].strip()) if match else None


def download_asset(offline_dir, filename):
    offline_dir.mkdir(parents=True, exist_ok=True)
    sums = offline_dir / "SHA256SUMS.txt"
    command = ["gh", "release", "download", RELEASE_TAG, "-R", REPO, "-p", filename, "-p", "SHA256SUMS.txt",
               "-D", str(offline_dir), "--clobber"]
    log("run: " + " ".join(command))
    subprocess.run(command, check=True)  # needs GH_TOKEN in the environment
    expected = parse_sha256sums(sums.read_text(encoding="utf-8")).get(filename)
    if not expected:
        raise SystemExit("SHA256SUMS.txt of release %s does not list %s" % (RELEASE_TAG, filename))
    actual = sha256_of(offline_dir / filename)
    if actual != expected:
        raise SystemExit("SHA-256 mismatch for %s: %s != %s" % (filename, actual, expected))
    log("verified %s (%d bytes, sha256 %s)" % (filename, (offline_dir / filename).stat().st_size, actual))
    return actual


def run_website_installer(python_slicer, slicer_exe, target_os, offline_dir, bundle_key, asset_sha):
    from shared.installer_script_builder import generate_installer_script, load_installer_artifacts

    artifacts = load_installer_artifacts()
    published = artifacts["bundles"][bundle_key].get("sha256", "")
    if published != asset_sha:
        raise SystemExit("docs/website/installer_artifacts.json lists sha256 %s for %s but the release asset is %s; "
                         "the website installer would download it again. Re-sync the checksums first." % (published, bundle_key, asset_sha))
    script = generate_installer_script(target_os=target_os, mode="base", slicer_path=str(slicer_exe), selected_packs=(),
                                       offline_dir=str(offline_dir), check_script_dir=False, artifacts=artifacts)
    script_path = offline_dir / ("Install-SlicerDSpace." + {"windows": "bat", "macos": "command", "linux": "sh"}[target_os])
    script_path.write_text(script, encoding="utf-8", newline="")
    env = dict(os.environ)
    env.update(SLICERDSPACE_RESOLVED_SLICER=str(slicer_exe), SLICERDSPACE_SCRIPT_DIR=str(offline_dir),
               SLICERDSPACE_INSTALLER_NONINTERACTIVE="1", PYTHONNOUSERSITE="1")
    command = [str(python_slicer), "-u", "-c", EXEC_SNIPPET, str(script_path)]
    log("run: %s -u -c <installer payload> %s" % (python_slicer, script_path))
    # cwd = the download folder, not the checkout. Run 44819974644: started inside the checkout, a fresh Slicer
    # reported SlicerDSpace as installed, the installer's own uninstall failed ("already loaded") and left it
    # scheduled for uninstall; the probe in run 44822858821 shows the cwd decides this.
    process = subprocess.Popen(command, env=env, cwd=str(offline_dir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, errors="replace")
    lines = []
    for line in process.stdout:
        lines.append(line)
        print(line, end="", flush=True)
    code = process.wait()
    if code != 0:
        raise SystemExit("The website installer exited with code %d (see its output above)" % code)
    return "".join(lines)


def overlay_checkout(module_dir):
    """Copy this checkout's runtime files over the installed module directory; verify the fingerprint."""
    from scripts.extension_runtime.runtime_resources import runtime_files
    from shared.code_updater import compute_code_tree_sha256

    files = runtime_files(REPO_ROOT)
    for relative in files:
        destination = module_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        shutil.copy2(str(REPO_ROOT / relative), str(destination))
    wanted, _ = compute_code_tree_sha256(REPO_ROOT)
    got, _ = compute_code_tree_sha256(module_dir)
    log("overlaid %d runtime files; fingerprint checkout=%s installed=%s" % (len(files), wanted[:16], got[:16]))
    if wanted != got:
        raise SystemExit("After the overlay the installed code tree differs from the checkout (fingerprint mismatch)")
    return len(files)


def strip_scheduled_uninstall(text, extension="SlicerDSpace"):
    """Remove one extension from the Qt ini key ScheduledForUninstall; return (new text, changed).

    Slicer keeps a list of extensions to delete at the next start in the settings key
    Extensions/ScheduledForUninstall. Slicer reads it before any script runs, so it cannot be cancelled
    from a Slicer script that starts afterwards; the ini file has to be edited first."""
    changed = False
    lines = []
    for line in text.splitlines(keepends=True):
        key, sep, value = line.partition("=")
        if sep and key.strip().replace("\\", "/").endswith("ScheduledForUninstall"):
            body = value.strip().strip('"')
            names = [n.strip() for n in body.split(",") if n.strip()]
            if extension in names:
                names = [n for n in names if n != extension]
                ending = line[len(line.rstrip("\r\n")):]
                line = "%s=%s%s" % (key, ",".join(names), ending)
                changed = True
        lines.append(line)
    return "".join(lines), changed


def settings_files_near(slicer_exe):
    """Qt ini files that Slicer may use for this install (next to the install, or in the user's profile)."""
    roots = [parent / "slicer.org" for parent in Path(slicer_exe).resolve().parents[:4]]
    home = Path.home()
    roots += [home / ".config" / "slicer.org", home / "Library" / "Preferences", home / "Library" / "Application Support" / "slicer.org"]
    for variable in ("APPDATA", "LOCALAPPDATA"):
        if os.environ.get(variable):
            roots += [Path(os.environ[variable]) / "slicer.org", Path(os.environ[variable]) / "NA-MIC"]
    found = []
    for root in roots:
        if root.is_dir():
            found += [p for p in root.glob("*.ini") if p not in found]
    return found


def scheduled_uninstall_files(slicer_exe, extension="SlicerDSpace"):
    """Settings files that list the extension under ScheduledForUninstall (Slicer deletes it at its next start)."""
    found = []
    for ini in settings_files_near(slicer_exe):
        text = ini.read_text(encoding="utf-8", errors="replace")
        if strip_scheduled_uninstall(text, extension)[1]:
            log("settings file %s: %s" % (ini, [l for l in text.splitlines() if "ScheduledForUninstall" in l]))
            found.append(ini)
    return found


def check_reinstall(installer_output, module_dir, slicer_exe):
    """Fail unless the second website-installer run replaced an installed copy and left nothing scheduled."""
    if REINSTALL_MARKER not in installer_output:
        raise SystemExit("The second install did not go over an installed SlicerDSpace (the installer never logged %r); "
                         "the reinstall path was not tested" % REINSTALL_MARKER)
    if module_dir is None or not (module_dir / "SlicerDSpace.py").is_file():
        raise SystemExit("SlicerDSpace is missing after the reinstall (module directory: %r)" % (module_dir,))
    leftovers = scheduled_uninstall_files(slicer_exe)
    if leftovers:
        print("::error title=Reinstall leaves SlicerDSpace scheduled for uninstall::%s list SlicerDSpace under "
              "Extensions/ScheduledForUninstall; the next Slicer start deletes the extension." % ", ".join(map(str, leftovers)), flush=True)
        raise SystemExit("After the reinstall %s still list SlicerDSpace under ScheduledForUninstall" % ", ".join(map(str, leftovers)))
    log("reinstall check passed: second install over an installed copy, module present, nothing scheduled for uninstall")


def cancel_scheduled_uninstall(slicer_exe):
    """Product-installer workaround for CI: if the website installer left SlicerDSpace scheduled for
    uninstall, the next Slicer start deletes the extension it just installed. Edit the ini files so the
    smoke test sees the installed layout, and print a GitHub warning with the evidence."""
    cancelled = []
    for ini in settings_files_near(slicer_exe):
        text = ini.read_text(encoding="utf-8", errors="replace")
        if "ScheduledForUninstall" in text:
            log("settings file %s: %s" % (ini, [l for l in text.splitlines() if "ScheduledForUninstall" in l]))
        new_text, changed = strip_scheduled_uninstall(text)
        if changed:
            ini.write_text(new_text, encoding="utf-8", newline="")
            cancelled.append(str(ini))
    if cancelled:
        print("::warning title=Product installer leaves SlicerDSpace scheduled for uninstall::"
              "After the website installer finished, %s listed SlicerDSpace under Extensions/ScheduledForUninstall, "
              "so the next Slicer start deletes the extension. CI cancelled the schedule to keep testing." % ", ".join(cancelled), flush=True)
    else:
        log("no ScheduledForUninstall entry for SlicerDSpace after the installer (nothing to cancel)")
    return cancelled


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
    parser.add_argument("--slicer-exe", required=True, type=Path)
    parser.add_argument("--python-slicer", required=True, type=Path)
    temp = Path(os.environ.get("RUNNER_TEMP") or os.environ.get("TMPDIR") or "/tmp")
    parser.add_argument("--offline-dir", type=Path, default=temp / "dspace-offline")
    parser.add_argument("--keep-asset", action="store_true", help="keep the downloaded base asset for a later --reinstall run")
    parser.add_argument("--reinstall", action="store_true",
                        help="run the website installer again over the installed copy and fail on any reinstall problem")
    args = parser.parse_args(argv)

    target_os, template, bundle_key = BASE_ASSETS[platform_key()]
    filename = template % RELEASE_TAG
    asset = args.offline_dir / filename
    if args.reinstall:
        if not asset.is_file():
            raise SystemExit("--reinstall needs %s from the first install run; pass --keep-asset to that run" % asset)
        asset_sha = sha256_of(asset)
    else:
        asset_sha = download_asset(args.offline_dir, filename)
    output = run_website_installer(args.python_slicer, args.slicer_exe, target_os, args.offline_dir, bundle_key, asset_sha)
    module_dir = find_installed_module_dir(output)
    if args.reinstall:
        check_reinstall(output, module_dir, args.slicer_exe)
    if module_dir is None or not (module_dir / "SlicerDSpace.py").is_file():
        raise SystemExit("Could not find the installed module directory in the installer output (got %r)" % (module_dir,))
    count = overlay_checkout(module_dir)
    cancelled = [] if args.reinstall else cancel_scheduled_uninstall(args.slicer_exe)
    if not args.keep_asset:
        asset.unlink()  # free about 1.5 GB of runner disk
    write_outputs({"module_dir": module_dir, "ext_root": module_dir.parents[2], "overlay_files": count,
                   "scheduled_uninstall_cancelled": len(cancelled)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
