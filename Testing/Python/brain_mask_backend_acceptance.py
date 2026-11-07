"""Opt-in real masking acceptance; never replaces a subject's supplied mask."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.brain_mask import BrainMaskConfig, generate_brain_mask


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("pnl_cnn",), required=True)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--script", type=Path)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--ants", type=Path)
    args = parser.parse_args()
    source, output = args.input_dir.resolve(), args.output_dir.resolve()
    if output == source or output.is_relative_to(source):
        parser.error("Acceptance output must be separate from the source input directory")
    output.mkdir(parents=True, exist_ok=True)
    supplied = source / "nodif_brain_mask.nii.gz"
    supplied_hash = digest(supplied) if supplied.exists() else None
    config = BrainMaskConfig(backend=args.backend, pnl_mask_python=args.python,
                             brain_mask_device=args.device,
                             pnl_mask_script=args.script, pnl_mask_model_dir=args.models,
                             pnl_mask_ants_bin_dir=args.ants)
    started = time.monotonic()
    report = {"backend": args.backend, "kind": "real_brain_mask_acceptance",
              "anatomical_review": "pending", "scope": "Real external inference and integrity, not accuracy certification"}
    try:
        result = generate_brain_mask(source / "data.nii.gz", source / "bvals", source / "bvecs",
                                     output / f"{args.backend}_mask.nii.gz", config=config,
                                     log=lambda message: print(message, flush=True))
        report.update(status="pass", result=result,
                      supplied_mask_unchanged=(digest(supplied) == supplied_hash) if supplied_hash else None)
        if report["supplied_mask_unchanged"] is False:
            raise RuntimeError("Supplied input mask changed")
    except Exception as exc:
        report.update(status="fail", error=str(exc))
    report["seconds"] = time.monotonic() - started
    (output / f"{args.backend}_acceptance.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "result"}, indent=2), flush=True)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
