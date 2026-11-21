"""Which algorithms this DATA cannot run (Qt-free).

The run planner stays the authority. This module reuses the planner's own data rules so that the GUI never greys
out or skips an algorithm that the planner would run:

* ``noddi``: ``microstructure.pipeline.noddi_nonzero_shells`` (same rounding and b0 threshold as
  ``shared.orchestrator._noddi_ineligible``): at least two non-zero b-shells.
* ``freewater`` and ``ifw``: ``microstructure.ifw.free_water_eligibility`` (DIPY ``check_multi_b``, same call as
  ``shared.orchestrator._free_water_ineligible``). For iFW only this b-value rule is a data block; its echo-time
  and upstream rules stay with the planner.
* ``dki``: ``microstructure.dki.dki_acquisition_problem``, the acquisition part of ``inspect_dki_inputs``. A
  missing MK-Curve backend or a bad option is not a data block (the environment check handles it and offers an
  install). The brain mask is not checked either.

Unknown data means available: no input, files that do not exist yet, unreadable b-values or gradients. Reading never
raises into the caller; any other failure counts as unknown and is logged once.

The functions at the end take the GUI widget as a duck-typed ``owner`` and import no Qt.
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

_LOG = logging.getLogger(__name__)
_LOGGED: set[tuple[str, str]] = set()
_CACHE: "OrderedDict[tuple, dict[str, Block]]" = OrderedDict()
_CACHE_LIMIT = 16
# F3: the per-event path (summary refresh, popup open) reuses the resolved gate files this long (seconds).
GATE_TTL_SECONDS = 1.0
_monotonic = time.monotonic

DATA_ALGORITHMS = ("noddi", "freewater", "ifw", "dki")


@dataclass(frozen=True)
class Block:
    """Why the data cannot run one algorithm: a stable ``code``, the numbers behind it, and the text."""

    algorithm: str
    code: str
    numbers: Mapping[str, Any] = field(default_factory=dict)

    def text(self, lang: str = "en") -> str:
        zh = str(lang or "").lower().startswith("zh")
        numbers = self.numbers
        if self.code == "noddi_needs_two_shells":
            found = ", ".join(f"{shell:g}" for shell in numbers["shells"]) or ("无" if zh else "none")
            if zh:
                return f"NODDI 至少需要 2 个非零 b 值壳层；当前数据有 {numbers['count']} 个（b={found}）。"
            return f"NODDI needs at least 2 non-zero b-shells; this DWI has {numbers['count']} (b={found})."
        if self.code == "free_water_needs_three_b_values":
            if zh:
                return "自由水模型至少需要三个不同的 b 值（b=0 也计入，例如 0、1000 和 2000 s/mm²）。"
            from microstructure.ifw import FREE_WATER_INELIGIBLE

            return FREE_WATER_INELIGIBLE
        if self.code == "dki_acquisition" or self.code in _DKI_ZH:
            if zh:
                # F5: the Chinese text comes from the code and its numbers, never from the English exception text.
                template = _DKI_ZH.get(self.code, "")
                if self.code == "dki_needs_4d_volumes" and numbers.get("volumes") is None:
                    template = _DKI_ZH_NOT_4D
                reason = template.format(**numbers) if template else ""
                return "当前数据不满足 DKI 的采集要求" + (f"：{reason}" if reason else "。")
            detail = str(numbers.get("detail", ""))
            return f"This DWI does not meet the DKI acquisition requirements: {detail}"
        return str(numbers.get("detail", self.code))


# F5: one Chinese text per DKI acquisition code of ``microstructure.dki`` (the English text is the planner's message).
_DKI_ZH = {
    "dki_needs_4d_volumes": "DKI 需要至少 {min_volumes} 个测量的 4D DWI，当前数据有 {volumes} 个。",
    "dki_affine_invalid": "DKI 需要有限且非奇异的 DWI 世界坐标仿射矩阵。",
    "dki_gradient_count_mismatch": "bvals/bvecs 必须与 DWI 体数一致：DWI 有 {volumes} 个体，bvals 有 {bvals} 个，bvecs 为 {bvecs}。",
    "dki_gradients_invalid": "梯度必须是有限值，且 b 值不能为负。",
    "dki_bvectors_not_unit": "扩散 b 向量必须是单位长度（允许误差 {tolerance:g}）。",
    "dki_needs_b0": "DKI 至少需要一个 b0 测量。",
    "dki_needs_two_shells": "按最大 b 值 {max_bvalue:g} 筛选后，DKI 至少需要 2 个非零壳层，当前有 {shells} 个。",
    "dki_shell_directions": ("DKI 每个壳层至少需要 {min_directions} 个独立方向且角度设计秩为 {min_rank}；"
                             "b={bvalue:g} 壳层有 {directions} 个方向，秩为 {rank}。"),
    "dki_design_rank": "DKI 缩放列设计矩阵的秩为 {rank}，{parameters} 个参数必须全部可识别。",
}
_DKI_ZH_NOT_4D = "DKI 需要 4D DWI，当前数据是 {dimensions}D。"


def clear_cache() -> None:
    """Forget cached results and logged failures (tests, or after a settings change)."""
    _CACHE.clear()
    _LOGGED.clear()


def _log_once(where: str, exc: BaseException) -> None:
    key = (where, f"{type(exc).__name__}: {exc}")
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    _LOG.warning("Algorithm availability check %s failed; treating the data as unknown: %s: %s",
                 where, type(exc).__name__, exc)


def _record(path: Any) -> Optional[tuple[str, int, int]]:
    """``(path, size, mtime_ns)`` of a regular file, else ``None``. One ``stat`` call."""
    if path is None:
        return None
    try:
        text = os.fspath(path)
        info = os.stat(text)
    except (OSError, TypeError, ValueError):
        return None
    import stat as stat_module

    if not stat_module.S_ISREG(info.st_mode):
        return None
    return (text, info.st_size, info.st_mtime_ns)


def _noddi_block(bvals) -> Optional[Block]:
    from microstructure.pipeline import MicrostructureConfig, noddi_nonzero_shells

    shells = noddi_nonzero_shells(bvals, MicrostructureConfig.noddi_b0_threshold)
    if len(shells) >= 2:
        return None
    return Block("noddi", "noddi_needs_two_shells", {"count": len(shells), "shells": list(shells)})


def _free_water_blocked(bvals) -> bool:
    from microstructure.ifw import free_water_eligibility

    return not free_water_eligibility(bvals)[0]


def _dki_block(dwi, bvals, bvec) -> Optional[Block]:
    import nibabel as nib
    import numpy as np

    image = nib.load(str(dwi))  # header only
    vectors = np.loadtxt(bvec, ndmin=2)
    from microstructure.dki import dki_acquisition_issue

    issue = dki_acquisition_issue(image, np.asarray(bvals, dtype=float).reshape(-1), vectors)
    if issue is None:
        return None
    # F5: the rule's own code and numbers; a ValueError without them (for example from DIPY) keeps the generic code.
    code = getattr(issue, "code", None)
    numbers = getattr(issue, "numbers", None)
    if not isinstance(code, str) or code not in _DKI_ZH or not isinstance(numbers, dict):
        code, numbers = "dki_acquisition", {}
    return Block("dki", code, {**numbers, "detail": str(issue)})


def _compute(dwi, bval, bvec, dwi_ok: bool, bvec_ok: bool) -> dict[str, Block]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # loadtxt on an empty file, DIPY on all-zero b-values
        return _compute_rules(dwi, bval, bvec, dwi_ok, bvec_ok)


def _compute_rules(dwi, bval, bvec, dwi_ok: bool, bvec_ok: bool) -> dict[str, Block]:
    import numpy as np

    try:
        bvals = np.loadtxt(bval, dtype=float).ravel()
    except ValueError:
        return {}  # an unreadable b-value file is reported by input validation
    blocks: dict[str, Block] = {}
    try:
        block = _noddi_block(bvals)
        if block is not None:
            blocks["noddi"] = block
    except Exception as exc:
        _log_once("NODDI", exc)
    try:
        if _free_water_blocked(bvals):
            code = "free_water_needs_three_b_values"
            blocks["freewater"] = Block("freewater", code)
            blocks["ifw"] = Block("ifw", code)
    except Exception as exc:
        _log_once("Free Water", exc)
    if dwi_ok and bvec_ok:
        try:
            block = _dki_block(dwi, bvals, bvec)
            if block is not None:
                blocks["dki"] = block
        except Exception as exc:
            _log_once("DKI", exc)
    return blocks


def data_blocks(inputs: Optional[Mapping[str, Any]]) -> dict[str, Block]:
    """Algorithm id -> why the DATA cannot run it. ``inputs`` has the planner's keys ``dwi``, ``bval``, ``bvec``.

    The result is cached on ``(path, size, mtime_ns)`` of the files read, so a repeat call does one ``stat`` per
    file. The returned dict is a copy.
    """
    try:
        if not inputs:
            return {}
        bval_record = _record(inputs.get("bval"))
        if bval_record is None:
            return {}  # no heavy import: the caller runs this on every parameter-node event
        dwi_record, bvec_record = _record(inputs.get("dwi")), _record(inputs.get("bvec"))
        key = (bval_record, dwi_record, bvec_record)
        cached = _CACHE.get(key)
        if cached is None:
            cached = _compute(
                dwi_record[0] if dwi_record else None, bval_record[0], bvec_record[0] if bvec_record else None,
                dwi_record is not None, bvec_record is not None,
            )
            _CACHE[key] = cached
            while len(_CACHE) > _CACHE_LIMIT:
                _CACHE.popitem(last=False)
        else:
            _CACHE.move_to_end(key)
        return dict(cached)
    except Exception as exc:  # never raise into the GUI
        _log_once("data", exc)
        return {}


# ---- GUI side: which files does the widget look at? (duck-typed owner, no Qt import) ----

def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _is_batch(owner: Any, node: Any) -> bool:
    check = getattr(owner, "_input_preview_is_batch", None)
    if callable(check):
        verdict = check()
        if isinstance(verdict, bool):
            return verdict
    cohort = getattr(node, "experimentsEnabled", False) is True and getattr(node, "experimentScope", "") == "cohort"
    return cohort or getattr(node, "batchMode", False) is True


def _sidecar(source: Path, suffix: str) -> Path:
    from shared.input_import import _sidecar_for

    return _sidecar_for(source, suffix)


def _import_pending(subject_root: Path, output_root: Optional[Path]) -> bool:
    """F2: True when the run would (re)import this subject before it reads ``input/``.

    ``run_subject_products`` re-imports a convertible source (``find_convertible_input``; its receipt test lives inside
    ``prepare_convertible_root`` and depends on the target workspace), and switches to a DICOM conversion whose
    receipt sits in a separate workspace (``prepare_dicom_root``). Either way the staged ``input/`` files may not be
    the ones that run, so the answer is unknown.
    """
    from shared.convertible_input import find_convertible_input

    if find_convertible_input(subject_root) is not None:
        return True
    if output_root is None:
        return False
    from shared.dicom_input import RECEIPT
    from shared.io_layout import subject_layout

    workspace = subject_layout(subject_root, output_root=output_root).workspace_root
    return workspace != subject_root and (workspace / "input" / RECEIPT).is_file()


def _planner_gate_inputs(owner: Any, root_text: str, subject_id: str, output_text: str,
                         fresh: bool) -> Optional[dict[str, str]]:
    """F1-F3: the files that the planner checks for this Run click, or ``None`` (unknown).

    The cached input inspection decides first: anything but a canonical subject root is unknown, with no folder
    listing. Then the roots are resolved the way ``_onRunSelectedClicked`` and ``plan_subject_products`` resolve them,
    and ``shared.orchestrator._resolve_analysis_inputs`` names the files (the final preprocessing files once
    ``_preprocessing_complete`` is true, else the raw ``input/`` files).
    """
    from shared.ui_input_inspection import inspect_input_directory

    inspected = inspect_input_directory(owner, Path(root_text).expanduser(), subject_id, refresh=fresh)
    if not isinstance(inspected, dict) or inspected.get("mode") != "subject_root":
        return None
    from shared.pipeline import resolve_subject_root_dir
    from shared.ui_run_controller import resolve_single_subject_output_root

    subject_root = Path(resolve_subject_root_dir(Path(root_text).expanduser(), subject_id)).expanduser().resolve()
    output_root = resolve_single_subject_output_root(output_text, subject_id)
    output_root = Path(output_root).expanduser().resolve() if output_root is not None else None
    if _import_pending(subject_root, output_root):
        return None
    from shared.orchestrator import _resolve_analysis_inputs

    inputs = _resolve_analysis_inputs(subject_root, subject_id, output_root)
    return {key: str(inputs[key]) for key in ("dwi", "bval", "bvec")}


def current_gate_inputs(owner: Any, *, fresh: bool = False) -> Optional[dict[str, str]]:
    """The ``dwi``/``bval``/``bvec`` files the next run would read, or ``None`` when they are unknown.

    1. A Prepare Inputs tile source that Run would import first: kind ``nifti`` uses the bval and bvec tiles, else
       the sidecars next to the NIfTI (the lookup of ``shared.input_import``). Kinds ``nrrd`` and ``dicom`` are
       unknown here.
    2. Else the files that the planner checks (``_planner_gate_inputs``). On the per-event path (``fresh=False``) the
       answer is reused for ``GATE_TTL_SECONDS`` per input folder, subject ID, output folder and tile selection; the
       Run filter passes ``fresh=True`` and always resolves again.
    3. Batch mode: ``None`` (subjects can differ).
    """

    try:
        node = getattr(owner, "_parameterNode", None)
        if node is None or _is_batch(owner, node):
            return None
        pending_source = getattr(owner, "_input_preview_pending_source", None)
        pending = pending_source() if callable(pending_source) else None
        if isinstance(pending, dict):
            source = _text(pending.get("source"))
            if pending.get("kind") != "nifti" or not source:
                return None
            tile = getattr(owner, "_prepare_inputs_widget_text", None)
            source_path = Path(source).expanduser()
            bval = _text(pending.get("bval")) or (_text(tile("bval")) if callable(tile) else "")
            bvec = _text(tile("bvec")) if callable(tile) else ""
            return {
                "dwi": str(source_path),
                "bval": bval or str(_sidecar(source_path, ".bval")),
                "bvec": bvec or str(_sidecar(source_path, ".bvec")),
            }
        root_text = _text(getattr(node, "inputDirectory", ""))
        if not root_text:
            return None
        subject_id = getattr(owner, "_effective_subject_id", None)
        subject_id = _text(subject_id() if callable(subject_id) else getattr(node, "subjectId", ""))
        output_text = _text(getattr(node, "outputDirectory", ""))
        key = (root_text, subject_id, output_text, repr(pending))
        now = _monotonic()
        cached = getattr(owner, "_algorithmGateCache", None)
        if (not fresh and isinstance(cached, tuple) and len(cached) == 3 and cached[0] == key
                and 0 <= now - cached[1] < GATE_TTL_SECONDS):
            return dict(cached[2]) if cached[2] is not None else None
        inputs = _planner_gate_inputs(owner, root_text, subject_id, output_text, fresh)
        try:
            owner._algorithmGateCache = (key, _monotonic(), inputs)
        except Exception:
            pass
        return dict(inputs) if inputs is not None else None
    except Exception as exc:
        _log_once("inputs", exc)
        return None


def current_data_blocks(owner: Any, *, fresh: bool = False) -> dict[str, Block]:
    """``data_blocks`` for the files the widget would run on; ``{}`` (everything available) when unknown.

    ``fresh=True`` (the Run filter) resolves the files again instead of reusing the short per-event cache.
    """
    return data_blocks(current_gate_inputs(owner, fresh=fresh))
