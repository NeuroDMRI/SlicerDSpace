"""Run official dwifslpreproc GPU-first without a CUDA-error CPU retry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import shlex
import subprocess
import sys


def configure_eddy(fsl, gpu_visible):
    original = fsl.eddy_binary
    cuda = original(True) if gpu_visible else ""
    cpu = original(False) if not cuda else ""
    # In the official CUDA-error handler, an empty CPU executable causes the
    # original error to propagate before any CPU correction is started.
    fsl.eddy_binary = lambda want_cuda: cuda if want_cuda else cpu
    return {"resolved": "cuda" if cuda else "cpu", "eddy_binary": cuda or cpu,
            "reason": "CUDA eddy available" if cuda else "No usable CUDA eddy in the selected external environment",
            "cuda_error_cpu_retry": False}


def normalize_eddy_threads(argv, resolved):
    """Official eddy_cuda accepts only --nthr=1; CPU keeps the user's count."""
    effective = list(argv)
    if resolved != "cuda" or "-eddy_options" not in effective:
        return effective
    index = effective.index("-eddy_options") + 1
    if index >= len(effective):
        raise ValueError("-eddy_options requires a value")
    tokens = shlex.split(effective[index])
    cleaned = []
    offset = 0
    while offset < len(tokens):
        if tokens[offset] == "--nthr":
            offset += 2
            continue
        if not tokens[offset].startswith("--nthr="):
            cleaned.append(tokens[offset])
        offset += 1
    effective[index] = shlex.join([*cleaned, "--nthr=1"])
    return effective


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("upstream_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    # Match the official script's own library-discovery path.
    sys.path.insert(0, str(args.script.parent.parent / "lib"))
    from mrtrix3 import app, fsl
    # fsl discovery can emit debug messages before upstream parses its CLI.
    app.verbosity = 1
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_visible = visible not in ("", "-1")
    if gpu_visible:
        try:
            smi = shutil.which("nvidia-smi") or (
                "/usr/lib/wsl/lib/nvidia-smi" if Path("/usr/lib/wsl/lib/nvidia-smi").is_file() else "nvidia-smi")
            no_win = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            probe = subprocess.run(
                [smi, "-L"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                creationflags=no_win,
            )
            gpu_visible = probe.returncode == 0 and "GPU " in probe.stdout
        except (OSError, subprocess.TimeoutExpired):
            if fsl.eddy_binary(True):
                raise RuntimeError("CUDA eddy is installed but GPU discovery failed; fix GPU access or explicitly hide CUDA devices")
            gpu_visible = False
    record = configure_eddy(fsl, gpu_visible)
    record["cuda_visible_devices"] = visible
    record["python_executable"] = sys.executable
    record["upstream_script_sha256"] = hashlib.sha256(args.script.read_bytes()).hexdigest()
    record["launcher_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.record.parent.mkdir(parents=True, exist_ok=True)
    upstream_args = args.upstream_args[1:] if args.upstream_args[:1] == ["--"] else args.upstream_args
    record["requested_upstream_argv"] = [str(args.script), *upstream_args]
    upstream_args = normalize_eddy_threads(upstream_args, record["resolved"])
    record["upstream_argv"] = [str(args.script), *upstream_args]
    args.record.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("FSL eddy device: " + json.dumps(record, sort_keys=True), flush=True)
    sys.argv = [str(args.script), *upstream_args]
    runpy.run_path(str(args.script), run_name="__main__")


if __name__ == "__main__":
    main()
