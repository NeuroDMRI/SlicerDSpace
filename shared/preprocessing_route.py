"""The one preprocessing route (W2a / D1), and the failure of the removed route flags.

Pure Python (no Slicer, Qt or heavy imports), so the CLI, the install scripts and the tests share it.
"""

from __future__ import annotations

import argparse
import os

ROUTE_ID = "dipy_mppca+gibbs+fsl_eddy"
CACHE_KEY_VERSION = 6


def resolve_preprocessing_execution_backend(selection: str | None = None) -> str:
    """Keep the validated runner as default while the shared native runner is qualified."""
    if selection is not None:
        choice = selection
    elif "DSP_PREPROCESSING_EXECUTION_BACKEND" in os.environ:
        choice = os.environ["DSP_PREPROCESSING_EXECUTION_BACKEND"]
    else:
        from shared.native_runtimes import installed_bundle_bin_dirs
        choice = "native" if installed_bundle_bin_dirs("fsl") else "legacy"
    choice = choice.strip().lower()
    if choice not in {"legacy", "native"}:
        raise ValueError("Preprocessing execution backend must be legacy or native.")
    return choice


def route_id(*, gibbs: bool = True, topup: bool = False) -> str:
    """W2b: the route id of one run: ``dipy_mppca+gibbs+fsl_eddy[+topup]``, without ``+gibbs`` when Gibbs removal is skipped."""
    return "dipy_mppca" + ("+gibbs" if gibbs else "") + "+fsl_eddy" + ("+topup" if topup else "")

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
