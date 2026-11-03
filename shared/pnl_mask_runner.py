"""Launch installed PNLBrainMask-Torch in the selected external Python.

No TensorFlow/Keras import, upstream AST repair, or Slicer Python installation.
The external console script owns inference and writes its full device receipt.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import sys


def select_device(requested: str, torch=None) -> dict:
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("PNL device must be auto, cpu or cuda")
    if requested == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return {"requested": requested, "resolved": "cpu", "framework": "pytorch",
                "reason": "Explicit CPU selection"}
    if torch is None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PNL requires PyTorch in the selected external Python") from exc
    available = torch.cuda.is_available()
    if requested == "cuda" and not available:
        raise RuntimeError("PNL GPU requested but external PyTorch has no visible CUDA GPU")
    if available:
        # Never unmask scheduler-restricted devices; force initialization to fail
        # here rather than silently accepting a broken CUDA runtime.
        torch.empty(1, device="cuda:0")
    return {"requested": requested, "resolved": "cuda" if available else "cpu",
            "framework": "pytorch", "torch_version": torch.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "reason": "First caller-visible CUDA GPU" if available else "No CUDA GPU visible to external PyTorch"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("upstream_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    record = {"requested": args.device, "framework": "pytorch"}
    try:
        record.update(select_device(args.device))
        args.record.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        script_args = args.upstream_args
        if script_args[:1] == ["--"]:
            script_args = script_args[1:]
        if script_args[:1] != ["mask"]:
            raise ValueError("PNL Torch launcher requires the standalone 'mask' subcommand")
        sys.argv = [str(args.script), *script_args, "--device", args.device, "--record", str(args.record)]
        runpy.run_path(str(args.script), run_name="__main__")
    except Exception as exc:
        record.update(status="blocked", error=str(exc))
        args.record.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
