#!/usr/bin/env python3
"""Read-only dependency check in a cancellable process, never the Qt thread."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.environment import run_environment_doctor, write_environment_report
from shared.environment_check import config_from_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    config = config_from_dict(json.loads(args.request.read_text(encoding='utf-8')))
    print('[Check] Checking selected dependencies; no installation or scientific processing.', flush=True)
    report = run_environment_doctor(
        config,
        progress_callback=lambda idx, total, label: print(f'[Check] ({idx}/{total}) {label}', flush=True),
    )
    write_environment_report(report, config.output_dir)
    temporary = args.result.with_suffix('.tmp')
    temporary.write_text(json.dumps(report.to_dict(), indent=2) + '\n', encoding='utf-8')
    temporary.replace(args.result)
    print(f'[Check] Complete: {report.blocker_count} required item(s) missing.', flush=True)


if __name__ == '__main__':
    main()
