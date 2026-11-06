#!/usr/bin/env python3
"""Run synthetic installation checks without loading patient data or changing packages."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_request(args):
    """The request file plus the per-feature options (B9): MRtrix, FSL, dcm2niix and the PNL native mask."""
    request = json.loads(args.request.read_text()) if args.request else {}
    if args.mrtrix_bin_dir:
        request['mrtrix_bin_dir'] = str(args.mrtrix_bin_dir)
    if args.fsl_bin_dir:
        request['fsl_bin_dir'] = str(args.fsl_bin_dir)
    if args.dcm2niix:
        request['dcm2niix'] = str(args.dcm2niix)
    if args.pnl_model_dir:
        request['pnl_native_mask'] = {'model_dir': str(args.pnl_model_dir), 'device': args.device}
    return request


def main(argv=None):
    from shared.setup_selftest import run_checks
    from shared.managed_runtimes import write_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, help='JSON request from the setup plan')
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--mrtrix-bin-dir', type=Path, help='Check mrconvert --version, dwi2response --help, dwifslpreproc --help')
    parser.add_argument('--fsl-bin-dir', type=Path, help='Check eddy, eddy_cpu and topup --help with the FSL environment')
    parser.add_argument('--dcm2niix', type=Path, help='Check dcm2niix -v')
    parser.add_argument('--pnl-model-dir', type=Path, help='Check the PNL native mask: three views, one slice each')
    parser.add_argument('--device', default='auto', choices=('auto', 'cpu', 'cuda'))
    args = parser.parse_args(argv)
    try:
        request = build_request(args)
        if not request:
            parser.error('give --request or at least one feature option')
        result = run_checks(request)
    except Exception as exc:
        result = dict(kind='slicerdspace_setup_selftest', status='failed', error=f'{type(exc).__name__}: {exc}')
    write_json(args.result, result)
    print(json.dumps(result, indent=2), flush=True)
    return int(result['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
