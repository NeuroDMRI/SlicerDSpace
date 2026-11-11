#!/usr/bin/env python3
"""Run author registration with explicitly audited bulk filtering and caching."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tractography.wma_registration import RegistrationPreprocessCache, registration_cache_support
from tractography.wma_preprocessing import AuthorCompatibleLengthFilter
from tractography.wma_transform import AuthorCompatibleFiberTransform, registration_transform_support
from tractography.wma_fibers import AuthorCompatibleFiberArrayConverter, fiber_conversion_support


def _run_native_nonrigid(author, author_args):
    from whitematteranalysis import register_two_subjects_nonrigid_bsplines as registration

    # The pinned author code references resource only in a progress message.
    if not hasattr(registration, "resource"):
        try:
            import resource
        except ImportError:
            from types import SimpleNamespace

            def getrusage(_):
                return SimpleNamespace(ru_maxrss="unavailable")

            resource = SimpleNamespace(RUSAGE_SELF=0, getrusage=getrusage)
        registration.resource = resource

    previous_argv = sys.argv
    try:
        sys.argv = [str(author), *author_args]
        runpy.run_path(str(author), run_name="__main__")
    finally:
        sys.argv = previous_argv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-script", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--native-nonrigid", action="store_true")
    parser.add_argument("author_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    author_args = args.author_args[1:] if args.author_args[:1] == ["--"] else args.author_args
    author = args.author_script.resolve(strict=True)
    if args.native_nonrigid:
        return _run_native_nonrigid(author, author_args)
    if args.report_dir is None:
        parser.error("--report-dir is required for cached registration.")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_dir / ("registration_cache_" + uuid.uuid4().hex + ".json")
    record = {"kind": "wma_registration_preprocess_cache", "version": 1,
              "author_script": str(author), "author_script_sha256": hashlib.sha256(author.read_bytes()).hexdigest(),
              "author_args": author_args, "status": "running", "enabled": False,
              "implementation_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (
                  Path(__file__), Path(__file__).resolve().parents[1] / 'tractography/wma_registration.py',
                  Path(__file__).resolve().parents[1] / 'tractography/wma_preprocessing.py')}}
    record['implementation_sha256']['wma_transform.py'] = hashlib.sha256(
        Path(__file__).resolve().parents[1].joinpath('tractography/wma_transform.py').read_bytes()).hexdigest()
    record['implementation_sha256']['wma_fibers.py'] = hashlib.sha256(
        Path(__file__).resolve().parents[1].joinpath('tractography/wma_fibers.py').read_bytes()).hexdigest()
    cache = original = filter_module = bulk_filter = None
    transform_module = original_transform = bulk_transform = None
    fiber_class = original_conversion = bulk_conversion = None
    previous_argv = sys.argv
    started = time.monotonic()
    try:
        import whitematteranalysis as wma
        package_dir = Path(wma.__file__).resolve().parent
        if (package_dir / "gpu" / "__init__.py").is_file():
            record["compatibility"] = {
                "supported": True,
                "native_wma_gpu": True,
                "reason": "whitematteranalysis_gpu provides built-in preprocess caching, bulk filtering, fast fiber conversion, and CPU/CUDA objectives",
            }
            record["enabled"] = True
        else:
            record["compatibility"] = registration_cache_support(package_dir)
            if record["compatibility"]["supported"]:
                filter_module = wma.filter
                original = filter_module.preprocess
                bulk_filter = AuthorCompatibleLengthFilter(original)
                cache = RegistrationPreprocessCache(bulk_filter)
                filter_module.preprocess = cache
                record["enabled"] = True
            record['transform_compatibility'] = registration_transform_support(package_dir)
            if record['transform_compatibility']['supported']:
                transform_module = wma.register_two_subjects
                original_transform = transform_module.transform_fiber_array_numpy
                bulk_transform = AuthorCompatibleFiberTransform(original_transform, transform_module.convert_transform_to_vtk)
                transform_module.transform_fiber_array_numpy = bulk_transform
            record['conversion_compatibility'] = fiber_conversion_support(package_dir)
            if record['conversion_compatibility']['supported']:
                fiber_class = wma.fibers.FiberArray
                original_conversion = fiber_class.convert_from_polydata
                bulk_conversion = AuthorCompatibleFiberArrayConverter(fiber_class)
                fiber_class.convert_from_polydata = bulk_conversion.method
        sys.argv = [str(author), *author_args]
        runpy.run_path(str(author), run_name="__main__")
        record["status"] = "completed"
    except SystemExit as exc:
        record["status"] = "completed" if exc.code in (None, 0) else "failed"
        record["exit_code"] = exc.code if isinstance(exc.code, (int, str)) else None
        raise
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        sys.argv = previous_argv
        if original is not None:
            filter_module.preprocess = original
        if original_transform is not None:
            transform_module.transform_fiber_array_numpy = original_transform
        if original_conversion is not None:
            fiber_class.convert_from_polydata = original_conversion
        if cache is not None:
            record["cache"] = cache.summary()
        if bulk_filter is not None:
            record["length_filter"] = bulk_filter.summary()
        if bulk_transform is not None:
            record['fiber_transform'] = bulk_transform.summary()
        if bulk_conversion is not None:
            record['fiber_array_conversion'] = bulk_conversion.summary()
        record["elapsed_seconds"] = time.monotonic() - started
        temporary = report_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(report_path)


if __name__ == "__main__":
    main()
