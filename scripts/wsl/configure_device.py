"""Choose and verify CPU/CUDA wheels in the isolated WSL runtime, not Slicer."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

TORCH_VERSION = "2.11.0"
TORCH_CUDA_DEFAULT_FLAVOR = "cu126"
TORCH_CUDA_BLACKWELL_FLAVOR = "cu128"
TORCH_INDEXES = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "cuda": "https://download.pytorch.org/whl/cu126",
    "cu126": "https://download.pytorch.org/whl/cu126",
    "cu128": "https://download.pytorch.org/whl/cu128",
}
FSL_CUDA_PACKAGE = "fsl-eddy-cuda=2602.0=h1a432fe_1"
_BLACKWELL_NAME_RE = re.compile(r"(\brtx\s*(?:pro\s*)?50\d\d\b|\bblackwell\b|\bg?b[12]00\b)", re.IGNORECASE)

_CUDA_SMOKE_PROBE = (
    "import json, sys, torch; "
    "import torch.nn.functional as F; "
    "cuda = bool(torch.cuda.is_available()); "
    "count = int(torch.cuda.device_count()) if cuda else 0; "
    "per_device = {}; "
    "kernel_ok = True; "
    "kernel_error = None; "
    "\n"
    "if sys.argv[1] == 'cuda' and cuda:\n"
    "    for _i in range(max(1, count)):\n"
    "        _dev = f'cuda:{_i}'\n"
    "        try:\n"
    "            _alloc = torch.empty(1, device=_dev)\n"
    "            _a = torch.ones((4, 4), dtype=torch.float32, device=_dev)\n"
    "            _b = torch.matmul(_a, _a)\n"
    "            _x = torch.ones((1, 1, 4, 4), dtype=torch.float32, device=_dev)\n"
    "            _w = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=_dev)\n"
    "            _c = torch.relu(F.conv2d(_x, _w))\n"
    "            torch.cuda.synchronize(_i)\n"
    "            _ok = bool(\n"
    "                abs(float(_b[0, 0].item()) - 4.0) < 1e-4\n"
    "                and abs(float(_c[0, 0, 0, 0].item()) - 9.0) < 1e-4\n"
    "            )\n"
    "            per_device[str(_i)] = {\n"
    "                'ok': _ok,\n"
    "                'name': torch.cuda.get_device_name(_i),\n"
    "                'capability': list(torch.cuda.get_device_capability(_i)),\n"
    "            }\n"
    "            if not _ok:\n"
    "                kernel_ok = False\n"
    "                kernel_error = f'CUDA smoke test numerical mismatch on {_dev}'\n"
    "        except Exception as _exc:\n"
    "            per_device[str(_i)] = {'ok': False, 'error': str(_exc)}\n"
    "            kernel_ok = False\n"
    "            kernel_error = str(_exc)\n"
    "print(json.dumps({\n"
    "    'version': torch.__version__,\n"
    "    'cuda_available': cuda,\n"
    "    'cuda_version': torch.version.cuda,\n"
    "    'device_count': count,\n"
    "    'kernel_ok': kernel_ok,\n"
    "    'kernel_error': kernel_error,\n"
    "    'per_device': per_device,\n"
    "}))\n"
)


def _find_nvidia_smi():
    command = shutil.which("nvidia-smi")
    if not command and Path("/usr/lib/wsl/lib/nvidia-smi").is_file():
        command = "/usr/lib/wsl/lib/nvidia-smi"
    return command


def _parse_compute_capability(text):
    raw = str(text or "").strip()
    if not raw or "." not in raw:
        return None
    parts = raw.split(".", 1)
    try:
        return (int(parts[0].strip()), int(parts[1].strip()))
    except ValueError:
        return None


def _filter_visible_gpus(gpus):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return gpus
    tokens = [tok.strip() for tok in str(visible).split(",") if tok.strip()]
    if not tokens or tokens == ["-1"]:
        return []
    selected = []
    for tok in tokens:
        if tok.isdigit():
            idx = int(tok)
            if 0 <= idx < len(gpus):
                selected.append(gpus[idx])
        else:
            for item in gpus:
                if tok in str(item.get("uuid") or "") or tok in str(item.get("raw") or ""):
                    selected.append(item)
                    break
    return selected or gpus


def discover_wsl_gpus():
    """Return visible GPU metadata (name, compute_capability) from ``nvidia-smi``, best-effort."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") in {"", "-1"}:
        return []
    command = _find_nvidia_smi()
    if not command:
        return []
    gpus = []
    try:
        query = subprocess.run(
            [command, "--query-gpu=name,compute_cap,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15,
        )
        if query.returncode == 0 and query.stdout.strip():
            for line in query.stdout.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if not parts or not parts[0]:
                    continue
                cap = _parse_compute_capability(parts[1]) if len(parts) > 1 else None
                uuid = parts[2] if len(parts) > 2 else ""
                gpus.append({
                    "name": parts[0],
                    "compute_capability": list(cap) if cap is not None else None,
                    "uuid": uuid,
                    "raw": line.strip(),
                })
    except Exception:
        pass
    if not gpus:
        try:
            listed = subprocess.run(
                [command, "-L"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15,
            )
            if listed.returncode == 0:
                for line in listed.stdout.splitlines():
                    if "GPU " in line:
                        name = line.split(":", 1)[1].split("(UUID:", 1)[0].strip() if ":" in line else line.strip()
                        gpus.append({"name": name, "compute_capability": None, "uuid": "", "raw": line.strip()})
        except Exception:
            pass
    return _filter_visible_gpus(gpus)


def requires_blackwell_cuda_wheel(*, gpus=None, compute_capabilities=None, gpu_names=None):
    """Return True when any target GPU is NVIDIA Blackwell (sm_100/sm_120, e.g. RTX 50 series)."""
    caps = list(compute_capabilities or ())
    names = list(gpu_names or ())
    for item in gpus or ():
        if isinstance(item, dict):
            if item.get("compute_capability"):
                caps.append(tuple(item["compute_capability"]))
            if item.get("name"):
                names.append(str(item["name"]))
    for cap in caps:
        try:
            major = int(cap[0])
        except (TypeError, ValueError, IndexError):
            continue
        if major >= 10:
            return True
    return any(_BLACKWELL_NAME_RE.search(str(name)) for name in names if name)


def select_cuda_flavor(*, gpus=None, compute_capabilities=None, gpu_names=None, cuda_flavor=None):
    """Pick the PyTorch CUDA wheel flavor (``cu128`` for Blackwell / RTX 50, ``cu126`` otherwise)."""
    override = str(cuda_flavor or os.environ.get("SLICERDSPACE_WSL_CUDA_FLAVOR") or "").strip().lower()
    if override:
        if override not in {"cu126", "cu128"}:
            raise ValueError(f"Unsupported WSL CUDA flavor {override!r}; expected 'cu126' or 'cu128'")
        return override
    if requires_blackwell_cuda_wheel(gpus=gpus, compute_capabilities=compute_capabilities, gpu_names=gpu_names):
        return TORCH_CUDA_BLACKWELL_FLAVOR
    return TORCH_CUDA_DEFAULT_FLAVOR


def cuda_available():
    if os.environ.get("CUDA_VISIBLE_DEVICES") in {"", "-1"}:
        return False
    command = _find_nvidia_smi()
    if not command:
        return False
    check = subprocess.run([command, "-L"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=15)
    if check.returncode:
        raise RuntimeError("WSL GPU discovery failed; fix the Windows NVIDIA/WSL driver or explicitly select CPU: " + check.stderr.strip())
    return "GPU " in check.stdout


def device_plan(requested="auto", gpu_available=None, *, compute_capabilities=None, gpu_names=None, cuda_flavor=None):
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("Device must be auto, cpu or cuda")
    detected = False if requested == "cpu" else cuda_available() if gpu_available is None else gpu_available
    if requested == "cuda" and not detected:
        raise RuntimeError("CUDA requested but no GPU is visible inside WSL; no CPU fallback")
    resolved = "cuda" if detected else "cpu"
    detected_gpus = []
    if resolved == "cuda" and gpu_available is None and compute_capabilities is None and gpu_names is None:
        detected_gpus = discover_wsl_gpus()
    if resolved == "cuda":
        flavor = select_cuda_flavor(
            gpus=detected_gpus,
            compute_capabilities=compute_capabilities,
            gpu_names=gpu_names,
            cuda_flavor=cuda_flavor,
        )
        torch_index = TORCH_INDEXES[flavor]
    else:
        flavor = "cpu"
        torch_index = TORCH_INDEXES["cpu"]
    return {
        "kind": "slicerdsurfer_wsl_device_plan",
        "status": "planned",
        "requested": requested,
        "resolved": resolved,
        "cuda_flavor": flavor,
        "torch_version": TORCH_VERSION,
        "torch_requirement": f"torch=={TORCH_VERSION}+{flavor}",
        "torch_index": torch_index,
        "fsl_cuda_package": FSL_CUDA_PACKAGE if resolved == "cuda" else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "detected_gpus": detected_gpus,
        "scientific_validation": "not_inferred_from_installation",
    }


def configure(plan):
    subprocess.run([sys.executable, "-m", "pip", "install", plan["torch_requirement"],
                    "--index-url", plan["torch_index"]], check=True)
    # Probe in a fresh interpreter after installation with real CUDA kernel ops
    # (matmul + conv2d + relu + synchronize + value check) so missing kernel images
    # (e.g. RTX 50 / sm_120 on a cu126 wheel) fail immediately during setup.
    probe = subprocess.run([sys.executable, "-c", _CUDA_SMOKE_PROBE, plan["resolved"]],
                           check=True, capture_output=True, text=True, timeout=90)
    actual = json.loads(probe.stdout)
    if actual["version"] != plan["torch_requirement"].split("==", 1)[1]:
        raise RuntimeError("WSL PyTorch wheel version differs from the selected device plan")
    if plan["resolved"] == "cuda":
        if not actual.get("cuda_available"):
            raise RuntimeError("CUDA wheel installed but GPU execution is unavailable; no CPU fallback")
        if actual.get("kernel_ok") is False:
            raise RuntimeError(
                "CUDA wheel installed but GPU kernel smoke test failed "
                "(likely missing kernel image for this GPU architecture, e.g. RTX 50 / sm_120 requiring cu128): "
                + str(actual.get("kernel_error") or "unknown kernel error")
            )
    return dict(plan, status="ready", actual_runtime=actual)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    try:
        plan = device_plan(args.device)
        result = configure(plan) if args.execute else plan
    except Exception as exc:
        result = {"kind": "slicerdsurfer_wsl_device_plan", "status": "failed",
                  "requested": args.device, "error": str(exc)}
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
