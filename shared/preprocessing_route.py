"""The one preprocessing route (W2a / D1), and the failure of the removed route flags.

Pure Python (no Slicer, Qt or heavy imports), so the CLI, the install scripts and the tests share it.
"""

from __future__ import annotations

import argparse

ROUTE_ID = "dipy_mppca+gibbs+fsl_eddy"
ROUTE_TEXT = "DIPY MP-PCA, DIPY Gibbs removal, FSL eddy/topup through dwifslpreproc"
# Flags of the removed route choice, the removed denoise choice and the removed between-volume motion correction.
REMOVED_FLAGS = ("--preprocessing-backend", "--preprocessing-dipy-denoise-method", "--preprocessing-skip-motion")


def removed_flag_message(flag: str) -> str:
    return f"Preprocessing has one route: {ROUTE_TEXT}; remove {flag}"


class RemovedPreprocessingFlag(argparse.Action):
    """An argparse action that fails with the message that names the single route."""

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        kwargs.setdefault("help", argparse.SUPPRESS)
        super().__init__(option_strings, dest, nargs="?" if nargs is None else nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        parser.error(removed_flag_message(option_string or self.option_strings[0]))


def add_removed_flags(parser: argparse.ArgumentParser, flags=REMOVED_FLAGS) -> None:
    """Register the removed flags only so that using one fails with the message. The valueless flag takes no value."""
    for flag in flags:
        parser.add_argument(flag, action=RemovedPreprocessingFlag, nargs=0 if flag.endswith("-motion") else "?")
