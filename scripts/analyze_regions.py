#!/usr/bin/env python3
"""Inspect a completed run or postprocess explicitly selected anatomical regions."""

import argparse
import json
from pathlib import Path
import re
import sys
import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.region_analysis import inspect_run, read_json, run_analysis, write_json


TRACT_RULES = ('through_any', 'through_all', 'endpoint_any', 'endpoint_both', 'endpoint_pair')


def _values(groups):
    values = [value.strip() for group in groups or [] for token in group
              for value in re.split(r'[,;，；]+', token) if value.strip()]
    return list(dict.fromkeys(values))


def main(argv=None):
    def cancelled(*_):
        raise KeyboardInterrupt("Regional job cancelled")

    signal.signal(signal.SIGTERM, cancelled)
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect", type=Path, help="One single-subject orchestration manifest")
    mode.add_argument("--request", type=Path, help="Regional analysis JSON request")
    mode.add_argument("--batch", type=Path, help="Shared regional request and completed subject manifests")
    mode.add_argument('--source-manifest', type=Path, nargs='+', action='append',
                      help='Completed subject run(s); repeat or give multiple paths for cohort export')
    parser.add_argument("--atlas", type=Path)
    parser.add_argument("--lut", type=Path)
    parser.add_argument("--catalog", type=Path, help='Optional JSON output for --inspect; otherwise print the catalog')
    direct = parser.add_argument_group('Direct regional analysis (no request JSON required)')
    direct.add_argument('--regions', nargs='+', action='append', help='Region names or IDs; spaces/commas and repeated flags supported')
    direct.add_argument('--metrics', nargs='+', action='append', help='Scalar metric keys, e.g. fa md noddi_ndi; defaults to FA when no analysis is selected')
    direct.add_argument('--output-dir', type=Path, help='Fresh result folder; cohort root when several sources are given')
    direct.add_argument('--tissue', action='store_true', help='Include DDSeg tissue composition')
    direct.add_argument('--surface', action='store_true', help='Include matching cortical morphology')
    direct.add_argument('--tractogram', help='Recorded tractography task key, e.g. tractography.ukf')
    direct.add_argument('--connectome', help='Recorded connectome task key, e.g. connectome.streamline_count')
    direct.add_argument('--tract-rule', choices=TRACT_RULES, default=None)
    for field in ('target', 'include', 'exclude'):
        direct.add_argument('--' + field + '-regions', nargs='+', action='append',
                            help='Region names/IDs for ' + field + ' selection; requires --tractogram')
    args = parser.parse_args(argv)
    direct_fields = ('regions', 'metrics', 'output_dir', 'tissue', 'surface', 'tractogram', 'connectome',
                     'tract_rule', 'target_regions', 'include_regions', 'exclude_regions')
    if not args.source_manifest and any(getattr(args, key) is not None and getattr(args, key) is not False
                                        for key in direct_fields):
        parser.error('Direct analysis options require --source-manifest; do not mix them with JSON requests or --inspect')
    if args.catalog and not args.inspect:
        parser.error('--catalog is only used with --inspect')
    if (args.request or args.batch) and (args.atlas or args.lut):
        parser.error('Put atlas/LUT overrides in the JSON, or use --source-manifest for direct options')
    if args.inspect:
        catalog = inspect_run(
                args.inspect.expanduser().resolve(),
                atlas_override=args.atlas.expanduser().resolve() if args.atlas else None,
                lut_path=args.lut.expanduser().resolve() if args.lut else None,
            )
        if args.catalog:
            write_json(args.catalog, catalog)
            print("[Regions] Available regions and QC-eligible inputs: " + str(args.catalog), flush=True)
        else:
            print(json.dumps(catalog, indent=2, ensure_ascii=False), flush=True)
    elif args.source_manifest:
        regions = _values(args.regions)
        if not regions or not args.output_dir:
            parser.error('--source-manifest requires --regions and --output-dir')
        if args.metrics is not None and not _values(args.metrics):
            parser.error('Metric lists must not be empty')
        if (args.target_regions or args.include_regions or args.exclude_regions or args.tract_rule) and not args.tractogram:
            parser.error('Tract rules and target/include/exclude regions require --tractogram')
        if args.tract_rule == 'endpoint_pair' and not _values(args.target_regions):
            parser.error('endpoint_pair requires --target-regions')
        sources = [str(path.expanduser().resolve()) for group in args.source_manifest for path in group]
        if len(set(sources)) != len(sources):
            parser.error('Duplicate --source-manifest paths are not allowed')
        metrics = _values(args.metrics) if args.metrics is not None else (
            [] if args.tissue or args.surface or args.tractogram or args.connectome else ['fa'])
        request = dict(selected_regions=regions, metrics=metrics, tissue=args.tissue, surface=args.surface,
                       tractogram=args.tractogram, connectome=args.connectome,
                       tract_rule=args.tract_rule or 'through_any')
        for field in ('target', 'include', 'exclude'):
            if getattr(args, field + '_regions') is not None:
                values = _values(getattr(args, field + '_regions'))
                if not values:
                    parser.error('--' + field + '-regions must not be empty')
                request[field + '_regions'] = values
        overrides = {}
        for option, key in (('atlas', 'atlas_override'), ('lut', 'lut_path')):
            if getattr(args, option):
                overrides[key] = str(getattr(args, option).expanduser().resolve())
        output = str(args.output_dir.expanduser().resolve())
        if len(sources) == 1:
            path = run_analysis(dict(request, source_manifest=sources[0], output_dir=output, **overrides))
            print('[Regions] Results: ' + str(path), flush=True)
        else:
            from shared.region_batch import run_region_batch

            path = run_region_batch(dict(request=request, subjects=[dict(source_manifest=path, **overrides)
                                                                   for path in sources], output_dir=output))
            print('[Regions] Cohort summaries: ' + str(path), flush=True)
            if read_json(path)['status'] != 'completed':
                raise SystemExit(1)
    elif args.batch:
        from shared.region_batch import run_region_batch

        path = run_region_batch(read_json(args.batch))
        print("[Regions] Cohort summaries: " + str(path), flush=True)
        if read_json(path)["status"] != "completed":
            raise SystemExit(1)
    else:
        run_analysis(read_json(args.request))


if __name__ == "__main__":
    main()
