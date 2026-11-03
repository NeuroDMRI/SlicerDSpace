#!/usr/bin/env python3
"""Whole-subject resource lease for matched sequential/subject benchmark arms."""

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.cohort_schedule import stage_admission


def main():
    arguments = sys.argv[1:]
    os.environ['SLICERDSPACE_COHORT_SUBJECT'] = arguments[arguments.index('--subject-id')+1]
    with stage_admission('whole_subject'):
        return subprocess.call([sys.executable, str(ROOT/'shared/run_subject_pipeline_cli.py'), *arguments])


if __name__ == '__main__':
    raise SystemExit(main())
