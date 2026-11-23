"""Validated parameter schemas and staged Qt settings panels for tractography and other algorithm pickers.

Mirrors the staged-commit contract of ``shared/ui_ukf.py``:
- Look-only popup clicks commit nothing (staged getter returns original untouched value).
- Invalid cross-field or out-of-range values raise ``ValueError`` during validation so Apply
  and outside-click keep the popup open and display the error.
- Empty/default JSON ``"{}"`` produces 100% backward-compatible backend configs and CLI arguments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Any, Mapping, Optional


# ---------------------------------------------------------------------------
# MRML-safe parameter node helpers (work on Slicer parameterNodeWrapper,
# raw MRML nodes, and SimpleNamespace test doubles)
# ---------------------------------------------------------------------------

def get_node_parameter(node: Any, name: str, default: Any = "") -> Any:
    """Read an attribute from a Slicer parameterNodeWrapper or test namespace."""

    if node is None:
        return default
    all_params = getattr(node, "allParameters", None)
    if isinstance(all_params, dict) and name in all_params:
        value = getattr(node, name, default)
        return default if value is None else value
    if getattr(node, "_is_parameter_node_wrapper", False):
        raw = getattr(node, "parameterNode", None)
        getter = getattr(raw, "GetParameter", None)
        if callable(getter):
            raw_val = getter(name)
            if isinstance(raw_val, str) and raw_val != "":
                return raw_val
    value = getattr(node, name, None)
    return default if value is None else value


def get_node_bool_parameter(node: Any, name: str, default: bool = True) -> bool:
    """Read a boolean parameter with MRML string fallback."""

    value = get_node_parameter(node, name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(default)


def set_node_parameter(node: Any, name: str, value: Any) -> None:
    """Write an attribute to a Slicer parameterNodeWrapper (persisting to MRML) or test namespace."""

    if node is None:
        return
    all_params = getattr(node, "allParameters", None)
    if isinstance(all_params, dict) and name in all_params:
        setattr(node, name, value)
        return
    if getattr(node, "_is_parameter_node_wrapper", False):
        raw = getattr(node, "parameterNode", None)
        setter = getattr(raw, "SetParameter", None)
        if callable(setter):
            raw_text = ("true" if value else "false") if isinstance(value, bool) else str(value)
            setter(name, raw_text)
    setattr(node, name, value)


# ---------------------------------------------------------------------------
# Pure Python validation helpers
# ---------------------------------------------------------------------------

def _load_json_mapping(raw: Mapping[str, Any] | str | None, label: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid {label} parameters JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"{label} parameters must be a JSON object.")
        return dict(parsed)
    if isinstance(raw, Mapping):
        return dict(raw)
    raise ValueError(f"{label} parameters must be a mapping or JSON string.")


def _check_known_keys(data: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown {label} parameter(s): {', '.join(unknown)}")


def _require_int(value: Any, name: str, minimum: int, maximum: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} {name} must be an integer.")
    if not (minimum <= value <= maximum):
        raise ValueError(f"{label} {name} must be between {minimum} and {maximum}; got {value}.")
    return int(value)


def _require_optional_pos_int(value: Any, name: str, maximum: int, label: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} {name} must be an integer or None.")
    if value == 0:
        return None
    if not (1 <= value <= maximum):
        raise ValueError(f"{label} {name} must be 0 (unseeded) or between 1 and {maximum}; got {value}.")
    return int(value)


def _require_float(value: Any, name: str, minimum: float, maximum: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} {name} must be a number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} {name} must be finite.")
    if not (minimum <= number <= maximum):
        raise ValueError(f"{label} {name} must be between {minimum:g} and {maximum:g}; got {number:g}.")
    return number


def _require_optional_pos_float(value: Any, name: str, maximum: float, label: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} {name} must be a number or None.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} {name} must be finite.")
    if number == 0.0:
        return None
    if not (0.0 < number <= maximum):
        raise ValueError(f"{label} {name} must be 0 (auto) or between 0 and {maximum:g}; got {number:g}.")
    return number


def _require_bool(value: Any, name: str, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} {name} must be a boolean.")
    return value


def _minimal_overrides_json(current: Mapping[str, Any], defaults: Mapping[str, Any]) -> str:
    diff = {key: value for key, value in current.items() if value != defaults[key]}
    return json.dumps(diff, sort_keys=True) if diff else "{}"


# ---------------------------------------------------------------------------
# 1. CSD / ACT tractography parameters
# ---------------------------------------------------------------------------

CSD_ACT_ALGORITHMS = ("iFOD2", "SD_Stream")
CSD_ACT_SEEDING_MODES = ("dynamic", "gmwmi")
CSD_ACT_KEYS = frozenset({
    "select_streamlines",
    "algorithm",
    "seeding",
    "min_length_mm",
    "max_length_mm",
    "step_size_mm",
    "cutoff",
    "use_sift2",
    "random_seed",
})


@dataclass(frozen=True)
class CSDActParameters:
    select_streamlines: int = 1_000_000
    algorithm: str = "iFOD2"
    seeding: str = "dynamic"
    min_length_mm: float = 10.0
    max_length_mm: float = 250.0
    step_size_mm: Optional[float] = None
    cutoff: Optional[float] = None
    use_sift2: bool = True
    random_seed: Optional[int] = None


CSD_ACT_DEFAULTS = asdict(CSDActParameters())


def _normalize_csd_act_entry(key: str, value: Any) -> Any:
    if key == "select_streamlines":
        return _require_int(value, key, 1000, 10_000_000, "CSD/ACT")
    if key == "algorithm":
        if not isinstance(value, str) or value not in CSD_ACT_ALGORITHMS:
            raise ValueError(f"CSD/ACT algorithm must be one of {', '.join(CSD_ACT_ALGORITHMS)}; got {value!r}.")
        return value
    if key == "seeding":
        if not isinstance(value, str) or value not in CSD_ACT_SEEDING_MODES:
            raise ValueError(f"CSD/ACT seeding must be one of {', '.join(CSD_ACT_SEEDING_MODES)}; got {value!r}.")
        return value
    if key == "min_length_mm":
        return _require_float(value, key, 1.0, 100.0, "CSD/ACT")
    if key == "max_length_mm":
        return _require_float(value, key, 10.0, 500.0, "CSD/ACT")
    if key == "step_size_mm":
        return _require_optional_pos_float(value, key, 5.0, "CSD/ACT")
    if key == "cutoff":
        return _require_optional_pos_float(value, key, 1.0, "CSD/ACT")
    if key == "use_sift2":
        return _require_bool(value, key, "CSD/ACT")
    if key == "random_seed":
        return _require_optional_pos_int(value, key, 2_147_483_647, "CSD/ACT")
    raise ValueError(f"Unknown CSD/ACT parameter: {key}")


def parse_csd_act_overrides(raw: Mapping[str, Any] | str | None = None) -> dict[str, Any]:
    """Validate and return only explicitly provided CSD/ACT override keys."""

    data = _load_json_mapping(raw, "CSD/ACT")
    _check_known_keys(data, set(CSD_ACT_KEYS), "CSD/ACT")
    overrides = {key: _normalize_csd_act_entry(key, value) for key, value in data.items()}
    effective = {**CSD_ACT_DEFAULTS, **overrides}
    if effective["min_length_mm"] >= effective["max_length_mm"]:
        raise ValueError(
            f"CSD/ACT requires min_length_mm ({effective['min_length_mm']:g}) "
            f"< max_length_mm ({effective['max_length_mm']:g})."
        )
    return overrides


def validate_csd_act_parameters(raw: Mapping[str, Any] | str | None = None) -> CSDActParameters:
    """Validate overrides and return a complete ``CSDActParameters`` instance."""

    overrides = parse_csd_act_overrides(raw)
    return CSDActParameters(**{**CSD_ACT_DEFAULTS, **overrides})


def csd_act_overrides_json(params: CSDActParameters) -> str:
    return _minimal_overrides_json(asdict(params), CSD_ACT_DEFAULTS)


# ---------------------------------------------------------------------------
# 2. DTI deterministic tractography parameters
# ---------------------------------------------------------------------------

DTI_FIT_METHODS = ("wls", "ols")
DTI_KEYS = frozenset({
    "fit_method",
    "step_size_mm",
    "max_angle_deg",
    "seed_density",
    "min_fa",
    "min_length_mm",
    "max_length_mm",
    "random_seed",
})


@dataclass(frozen=True)
class DTIParameters:
    fit_method: str = "wls"
    step_size_mm: float = 0.5
    max_angle_deg: float = 30.0
    seed_density: int = 1
    min_fa: float = 0.2
    min_length_mm: float = 10.0
    max_length_mm: float = 250.0
    random_seed: int = 12345


DTI_DEFAULTS = asdict(DTIParameters())


def dti_backend_fit_method(fit_method: str) -> str:
    """Map UI/JSON DTI fit method ('wls', 'ols', 'ls') to ``DTITrackingConfig.fit_method`` ('wls' or 'ls')."""

    normalized = str(fit_method or "wls").strip().lower()
    if normalized == "wls":
        return "wls"
    if normalized in {"ols", "ls"}:
        return "ls"
    raise ValueError(f"DTI fit_method must be 'wls' or 'ols'; got {fit_method!r}.")


def _normalize_dti_entry(key: str, value: Any) -> Any:
    if key == "fit_method":
        if not isinstance(value, str):
            raise ValueError(f"DTI fit_method must be 'wls' or 'ols'; got {value!r}.")
        lowered = value.strip().lower()
        if lowered == "wls":
            return "wls"
        if lowered in {"ols", "ls"}:
            return "ols"
        raise ValueError(f"DTI fit_method must be 'wls' or 'ols'; got {value!r}.")
    if key == "step_size_mm":
        return _require_float(value, key, 0.1, 5.0, "DTI")
    if key == "max_angle_deg":
        return _require_float(value, key, 5.0, 90.0, "DTI")
    if key == "seed_density":
        return _require_int(value, key, 1, 5, "DTI")
    if key == "min_fa":
        return _require_float(value, key, 0.01, 0.95, "DTI")
    if key == "min_length_mm":
        return _require_float(value, key, 1.0, 100.0, "DTI")
    if key == "max_length_mm":
        return _require_float(value, key, 10.0, 500.0, "DTI")
    if key == "random_seed":
        return _require_int(value, key, 0, 2_147_483_647, "DTI")
    raise ValueError(f"Unknown DTI parameter: {key}")


def parse_dti_overrides(raw: Mapping[str, Any] | str | None = None) -> dict[str, Any]:
    """Validate and return only explicitly provided DTI tracking override keys."""

    data = _load_json_mapping(raw, "DTI")
    _check_known_keys(data, set(DTI_KEYS), "DTI")
    overrides = {key: _normalize_dti_entry(key, value) for key, value in data.items()}
    effective = {**DTI_DEFAULTS, **overrides}
    if effective["min_length_mm"] >= effective["max_length_mm"]:
        raise ValueError(
            f"DTI requires min_length_mm ({effective['min_length_mm']:g}) "
            f"< max_length_mm ({effective['max_length_mm']:g})."
        )
    return overrides


def validate_dti_parameters(raw: Mapping[str, Any] | str | None = None) -> DTIParameters:
    overrides = parse_dti_overrides(raw)
    return DTIParameters(**{**DTI_DEFAULTS, **overrides})


def dti_overrides_json(params: DTIParameters) -> str:
    return _minimal_overrides_json(asdict(params), DTI_DEFAULTS)


# ---------------------------------------------------------------------------
# 3. SAF short association fibre parameters
# ---------------------------------------------------------------------------

SAF_KEYS = frozenset({
    "select_streamlines",
    "maxlen",
    "notrunc",
    "fast_intersection",
    "gwg_hard",
    "pialfilter",
    "random_seed",
})


@dataclass(frozen=True)
class SAFParameters:
    select_streamlines: int = 1_000_000
    maxlen: float = 90.0
    notrunc: bool = False
    fast_intersection: bool = True
    gwg_hard: bool = False
    pialfilter: bool = False
    random_seed: int = 12345


SAF_DEFAULTS = asdict(SAFParameters())


def _normalize_saf_entry(key: str, value: Any) -> Any:
    if key == "select_streamlines":
        return _require_int(value, key, 1000, 10_000_000, "SAF")
    if key == "maxlen":
        return _require_float(value, key, 10.0, 300.0, "SAF")
    if key in {"notrunc", "fast_intersection", "gwg_hard", "pialfilter"}:
        return _require_bool(value, key, "SAF")
    if key == "random_seed":
        return _require_int(value, key, 0, 2_147_483_647, "SAF")
    raise ValueError(f"Unknown SAF parameter: {key}")


def parse_saf_overrides(raw: Mapping[str, Any] | str | None = None) -> dict[str, Any]:
    """Validate and return only explicitly provided SAF override keys."""

    data = _load_json_mapping(raw, "SAF")
    _check_known_keys(data, set(SAF_KEYS), "SAF")
    return {key: _normalize_saf_entry(key, value) for key, value in data.items()}


def validate_saf_parameters(raw: Mapping[str, Any] | str | None = None) -> SAFParameters:
    overrides = parse_saf_overrides(raw)
    return SAFParameters(**{**SAF_DEFAULTS, **overrides})


def saf_overrides_json(params: SAFParameters) -> str:
    return _minimal_overrides_json(asdict(params), SAF_DEFAULTS)


# ---------------------------------------------------------------------------
# 4. Audited non-tractography algorithm parameters
# ---------------------------------------------------------------------------

def validate_microstructure_dti_fit_method(value: Any) -> str:
    """Validate microstructure DTI tensor fit method ('wls' or 'ls'; accepts 'ols' alias)."""

    if not isinstance(value, str):
        raise ValueError(f"Microstructure DTI fit_method must be 'wls' or 'ls'; got {value!r}.")
    lowered = value.strip().lower()
    if lowered == "wls":
        return "wls"
    if lowered in {"ls", "ols"}:
        return "ls"
    raise ValueError(f"Microstructure DTI fit_method must be 'wls' or 'ls'; got {value!r}.")


FREEWATER_DEFAULTS = {"max_fraction_for_tissue": 0.7}


def parse_freewater_overrides(raw: Mapping[str, Any] | str | None = None) -> dict[str, Any]:
    data = _load_json_mapping(raw, "Free Water")
    _check_known_keys(data, set(FREEWATER_DEFAULTS), "Free Water")
    overrides: dict[str, Any] = {}
    if "max_fraction_for_tissue" in data:
        overrides["max_fraction_for_tissue"] = _require_float(
            data["max_fraction_for_tissue"], "max_fraction_for_tissue", 0.05, 0.99, "Free Water",
        )
    return overrides


IFW_T2_DEFAULTS = {"t2_gm_ms": 95.0, "t2_csf_ms": 1500.0}


def parse_ifw_overrides(raw: Mapping[str, Any] | str | None = None) -> dict[str, Any]:
    data = _load_json_mapping(raw, "iFW")
    _check_known_keys(data, set(IFW_T2_DEFAULTS), "iFW")
    overrides: dict[str, Any] = {}
    for key in ("t2_gm_ms", "t2_csf_ms"):
        if key in data:
            overrides[key] = _require_float(data[key], key, 1.0, 5000.0, "iFW")
    return overrides


TRACTCLOUD_DEFAULTS = {"batch_size": 2048}


def parse_tractcloud_overrides(raw: Mapping[str, Any] | str | None = None) -> dict[str, Any]:
    data = _load_json_mapping(raw, "TractCloud")
    _check_known_keys(data, set(TRACTCLOUD_DEFAULTS), "TractCloud")
    overrides: dict[str, Any] = {}
    if "batch_size" in data:
        overrides["batch_size"] = _require_int(data["batch_size"], "batch_size", 64, 16384, "TractCloud")
    return overrides


# ---------------------------------------------------------------------------
# Qt control helpers (safe with both real Slicer PythonQt and fake_qt doubles)
# ---------------------------------------------------------------------------

def _read_spin(control: Any, *, as_int: bool = False, decimals: int = 3) -> int | float:
    raw = control.value() if callable(getattr(control, "value", None)) else control.value
    return int(raw) if as_int else round(float(raw), decimals)


def _read_check(control: Any) -> bool:
    raw = control.checked() if callable(getattr(control, "checked", None)) else control.checked
    return bool(raw)


def _make_combo(qt: Any, entries: tuple[tuple[str, str, str], ...], zh: bool, obj_name: str, *, dark: Any = None):
    combo = qt.QComboBox()
    combo.setObjectName(obj_name)
    valid = [code for code, _en, _zh_label in entries]
    for code, en_label, zh_label in entries:
        combo.addItem(zh_label if zh else en_label, code)
    try:
        from shared.ui_styles import style_combobox_popup

        style_combobox_popup(combo, dark=dark)
    except Exception:
        pass
    tracked = {"value": valid[0]}

    def set_code(code: str) -> None:
        idx = valid.index(code) if code in valid else 0
        tracked["value"] = valid[idx]
        combo.setCurrentIndex(idx)

    def get_code() -> str:
        raw_idx = getattr(combo, "currentIndex", None)
        idx = raw_idx() if callable(raw_idx) else raw_idx
        if isinstance(idx, int) and 0 <= idx < len(valid):
            data = combo.itemData(idx)
            if isinstance(data, str) and data in valid:
                return data
            return valid[idx]
        data = combo.itemData(idx)
        if isinstance(data, str) and data in valid:
            return data
        return tracked["value"]

    return combo, set_code, get_code


def _safe_load_overrides(raw: str, parser) -> dict[str, Any]:
    """Load persisted overrides for UI display; fall back to defaults if stored JSON is corrupted."""

    try:
        return dict(parser(raw))
    except ValueError:
        return {}


def _add_restore_button(qt: Any, layout: Any, zh: bool, obj_name: str, on_reset) -> Any:
    button = qt.QPushButton("恢复默认值" if zh else "Restore defaults")
    button.setObjectName(obj_name)
    if callable(getattr(button, "setAutoDefault", None)):
        button.setAutoDefault(False)
    button.clicked.connect(lambda _checked=False: on_reset())
    layout.addRow(button)
    return button


# ---------------------------------------------------------------------------
# Staged Qt panels — Tractography
# ---------------------------------------------------------------------------

CSD_ACT_ALGORITHM_ITEMS = (
    ("iFOD2", "iFOD2 (probabilistic)", "iFOD2（概率追踪）"),
    ("SD_Stream", "SD_Stream (deterministic)", "SD_Stream（确定性追踪）"),
)

CSD_ACT_SEEDING_ITEMS = (
    ("dynamic", "Dynamic (WM FOD)", "动态播种 (WM FOD)"),
    ("gmwmi", "GM–WM interface (GMWMI)", "灰白质交界 (GMWMI)"),
)


def add_staged_csd_act_fields(layout: Any, node: Any, zh: bool, *, dark: Any = None):
    """Build staged CSD/ACT parameter controls; returns ``(staged_json_fn, staged_streamlines_fn | None)``."""

    import qt

    controls: dict[str, Any] = {}
    combo_setters: dict[str, Any] = {}
    combo_getters: dict[str, Any] = {}

    streamlines_spin = qt.QSpinBox()
    streamlines_spin.setRange(1000, 10_000_000)
    streamlines_spin.setSingleStep(50_000)
    streamlines_spin.setObjectName("csd_act_select_streamlines")
    layout.addRow("纤维条数" if zh else "Streamlines", streamlines_spin)
    controls["select_streamlines"] = streamlines_spin

    alg_combo, set_alg, get_alg = _make_combo(qt, CSD_ACT_ALGORITHM_ITEMS, zh, "csd_act_algorithm", dark=dark)
    layout.addRow("追踪算法" if zh else "Tracking algorithm", alg_combo)
    controls["algorithm"] = alg_combo
    combo_setters["algorithm"] = set_alg
    combo_getters["algorithm"] = get_alg

    seed_combo, set_seed, get_seed = _make_combo(qt, CSD_ACT_SEEDING_ITEMS, zh, "csd_act_seeding", dark=dark)
    layout.addRow("播种策略" if zh else "Seeding mode", seed_combo)
    controls["seeding"] = seed_combo
    combo_setters["seeding"] = set_seed
    combo_getters["seeding"] = get_seed

    float_specs = (
        ("min_length_mm", "Min length (mm)", "最小长度 (mm)", 1.0, 100.0, 1.0, 1),
        ("max_length_mm", "Max length (mm)", "最大长度 (mm)", 10.0, 500.0, 10.0, 1),
        ("step_size_mm", "Step size (mm, 0 = auto)", "步长 (mm，0 = 自动)", 0.0, 5.0, 0.1, 2),
        ("cutoff", "FOD cutoff (0 = auto)", "FOD 阈值 (0 = 自动)", 0.0, 1.0, 0.01, 3),
    )
    decimals_map = {key: dec for key, *_rest, dec in float_specs}
    for key, en, chinese, minimum, maximum, step, decimals in float_specs:
        spin = qt.QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setObjectName("csd_act_" + key)
        layout.addRow(chinese if zh else en, spin)
        controls[key] = spin

    sift2_box = qt.QCheckBox()
    sift2_box.setObjectName("csd_act_use_sift2")
    layout.addRow("启用 SIFT2 权重" if zh else "Run SIFT2 weighting", sift2_box)
    controls["use_sift2"] = sift2_box

    rng_spin = qt.QSpinBox()
    rng_spin.setRange(0, 2_147_483_647)
    rng_spin.setSingleStep(1)
    rng_spin.setObjectName("csd_act_random_seed")
    layout.addRow("随机种子 (0 = 随机)" if zh else "Random seed (0 = unseeded)", rng_spin)
    controls["random_seed"] = rng_spin

    def show_values(raw_values: Mapping[str, Any]) -> None:
        validated = asdict(validate_csd_act_parameters(raw_values))
        controls["select_streamlines"].setValue(int(validated["select_streamlines"]))
        combo_setters["algorithm"](validated["algorithm"])
        combo_setters["seeding"](validated["seeding"])
        controls["min_length_mm"].setValue(float(validated["min_length_mm"]))
        controls["max_length_mm"].setValue(float(validated["max_length_mm"]))
        controls["step_size_mm"].setValue(0.0 if validated["step_size_mm"] is None else float(validated["step_size_mm"]))
        controls["cutoff"].setValue(0.0 if validated["cutoff"] is None else float(validated["cutoff"]))
        controls["use_sift2"].setChecked(bool(validated["use_sift2"]))
        controls["random_seed"].setValue(0 if validated["random_seed"] is None else int(validated["random_seed"]))

    def read_validated() -> CSDActParameters:
        step_val = _read_spin(controls["step_size_mm"], decimals=decimals_map["step_size_mm"])
        cutoff_val = _read_spin(controls["cutoff"], decimals=decimals_map["cutoff"])
        seed_val = _read_spin(controls["random_seed"], as_int=True)
        values = {
            "select_streamlines": _read_spin(controls["select_streamlines"], as_int=True),
            "algorithm": combo_getters["algorithm"](),
            "seeding": combo_getters["seeding"](),
            "min_length_mm": _read_spin(controls["min_length_mm"], decimals=decimals_map["min_length_mm"]),
            "max_length_mm": _read_spin(controls["max_length_mm"], decimals=decimals_map["max_length_mm"]),
            "step_size_mm": None if step_val == 0.0 else step_val,
            "cutoff": None if cutoff_val == 0.0 else cutoff_val,
            "use_sift2": _read_check(controls["use_sift2"]),
            "random_seed": None if seed_val == 0 else seed_val,
        }
        return validate_csd_act_parameters(values)

    original = str(get_node_parameter(node, "csdActParameters", "{}") or "{}").strip() or "{}"
    saved_dict = _safe_load_overrides(original, parse_csd_act_overrides)
    has_streamlines_attr = node is not None and (
        hasattr(node, "orchestrationSelectStreamlines")
        or "orchestrationSelectStreamlines" in (getattr(node, "allParameters", None) or {})
    )
    original_streamlines = (
        int(get_node_parameter(node, "orchestrationSelectStreamlines", 1_000_000) or 1_000_000)
        if has_streamlines_attr
        else None
    )
    display_dict = dict(saved_dict)
    if (
        "select_streamlines" not in display_dict
        and original_streamlines is not None
        and 1000 <= original_streamlines <= 10_000_000
    ):
        display_dict["select_streamlines"] = original_streamlines
    show_values(display_dict)
    initial = read_validated()

    def staged_json() -> str:
        current = read_validated()
        return original if current == initial else csd_act_overrides_json(current)

    staged_streamlines = None
    if has_streamlines_attr:
        def staged_streamlines_fn() -> int:
            current = read_validated()
            if current.select_streamlines == initial.select_streamlines:
                return int(original_streamlines)
            return int(current.select_streamlines)

        staged_streamlines = staged_streamlines_fn

    _add_restore_button(qt, layout, zh, "csd_act_restore_defaults", lambda: show_values({}))
    return staged_json, staged_streamlines


DTI_FIT_METHOD_ITEMS = (
    ("wls", "Weighted least squares (WLS)", "加权最小二乘 (WLS)"),
    ("ols", "Ordinary least squares (OLS)", "普通最小二乘 (OLS)"),
)


def add_staged_dti_det_fields(layout: Any, node: Any, zh: bool, *, dark: Any = None):
    """Build staged DTI deterministic tractography parameter controls."""

    import qt

    controls: dict[str, Any] = {}
    fit_combo, set_fit, get_fit = _make_combo(qt, DTI_FIT_METHOD_ITEMS, zh, "dti_det_fit_method", dark=dark)
    layout.addRow("张量拟合方法" if zh else "Tensor fit method", fit_combo)
    controls["fit_method"] = fit_combo

    specs = (
        ("step_size_mm", "Tracking step (mm)", "追踪步长 (mm)", 0.1, 5.0, 0.1, 2),
        ("max_angle_deg", "Max turning angle (°)", "最大转角 (°)", 5.0, 90.0, 5.0, 1),
        ("seed_density", "Seed density per axis", "每轴播种密度", 1, 5, 1, 0),
        ("min_fa", "Minimum FA threshold", "最小 FA 阈值", 0.01, 0.95, 0.05, 2),
        ("min_length_mm", "Min length (mm)", "最小长度 (mm)", 1.0, 100.0, 1.0, 1),
        ("max_length_mm", "Max length (mm)", "最大长度 (mm)", 10.0, 500.0, 10.0, 1),
        ("random_seed", "Random seed", "随机种子", 0, 2_147_483_647, 1, 0),
    )
    decimals_map = {key: dec for key, *_rest, dec in specs}
    int_keys = {key for key, _en, _ch, minimum, *_rest in specs if isinstance(minimum, int)}
    for key, en, chinese, minimum, maximum, step, decimals in specs:
        if isinstance(minimum, int):
            control = qt.QSpinBox()
        else:
            control = qt.QDoubleSpinBox()
            control.setDecimals(decimals)
        control.setRange(minimum, maximum)
        control.setSingleStep(step)
        control.setObjectName("dti_det_" + key)
        layout.addRow(chinese if zh else en, control)
        controls[key] = control

    def show_values(raw_values: Mapping[str, Any]) -> None:
        validated = asdict(validate_dti_parameters(raw_values))
        set_fit(validated["fit_method"])
        for key in decimals_map:
            controls[key].setValue(validated[key])

    def read_validated() -> DTIParameters:
        values: dict[str, Any] = {"fit_method": get_fit()}
        for key, dec in decimals_map.items():
            values[key] = _read_spin(controls[key], as_int=(key in int_keys), decimals=dec)
        return validate_dti_parameters(values)

    original = str(get_node_parameter(node, "dtiParameters", "{}") or "{}").strip() or "{}"
    show_values(_safe_load_overrides(original, parse_dti_overrides))
    initial = read_validated()

    def staged_value() -> str:
        current = read_validated()
        return original if current == initial else dti_overrides_json(current)

    _add_restore_button(qt, layout, zh, "dti_det_restore_defaults", lambda: show_values({}))
    return staged_value


def add_staged_saf_fields(layout: Any, node: Any, zh: bool):
    """Build staged SAF short association fibre parameter controls."""

    import qt

    controls: dict[str, Any] = {}

    streamlines_spin = qt.QSpinBox()
    streamlines_spin.setRange(1000, 10_000_000)
    streamlines_spin.setSingleStep(50_000)
    streamlines_spin.setObjectName("saf_select_streamlines")
    layout.addRow("初始纤维条数" if zh else "Initial streamlines", streamlines_spin)
    controls["select_streamlines"] = streamlines_spin

    maxlen_spin = qt.QDoubleSpinBox()
    maxlen_spin.setRange(10.0, 300.0)
    maxlen_spin.setSingleStep(5.0)
    maxlen_spin.setDecimals(1)
    maxlen_spin.setObjectName("saf_maxlen")
    layout.addRow("最大短纤维长度 (mm)" if zh else "Max SAF length (mm)", maxlen_spin)
    controls["maxlen"] = maxlen_spin

    bool_specs = (
        ("notrunc", "Keep untruncated segments", "保留未截断皮层段"),
        ("fast_intersection", "Fast surface intersection", "快速表面相交检测"),
        ("gwg_hard", "Strict GM–WM–GM filter", "严格灰质-白质-灰质过滤"),
        ("pialfilter", "Filter pial-crossing fibres", "过滤穿越软脑膜纤维"),
    )
    for key, en, chinese in bool_specs:
        box = qt.QCheckBox()
        box.setObjectName("saf_" + key)
        layout.addRow(chinese if zh else en, box)
        controls[key] = box

    seed_spin = qt.QSpinBox()
    seed_spin.setRange(0, 2_147_483_647)
    seed_spin.setSingleStep(1)
    seed_spin.setObjectName("saf_random_seed")
    layout.addRow("随机种子" if zh else "Random seed", seed_spin)
    controls["random_seed"] = seed_spin

    def show_values(raw_values: Mapping[str, Any]) -> None:
        validated = asdict(validate_saf_parameters(raw_values))
        controls["select_streamlines"].setValue(int(validated["select_streamlines"]))
        controls["maxlen"].setValue(float(validated["maxlen"]))
        for key, _en, _ch in bool_specs:
            controls[key].setChecked(bool(validated[key]))
        controls["random_seed"].setValue(int(validated["random_seed"]))

    def read_validated() -> SAFParameters:
        values = {
            "select_streamlines": _read_spin(controls["select_streamlines"], as_int=True),
            "maxlen": _read_spin(controls["maxlen"], decimals=1),
            "notrunc": _read_check(controls["notrunc"]),
            "fast_intersection": _read_check(controls["fast_intersection"]),
            "gwg_hard": _read_check(controls["gwg_hard"]),
            "pialfilter": _read_check(controls["pialfilter"]),
            "random_seed": _read_spin(controls["random_seed"], as_int=True),
        }
        return validate_saf_parameters(values)

    original = str(get_node_parameter(node, "safParameters", "{}") or "{}").strip() or "{}"
    show_values(_safe_load_overrides(original, parse_saf_overrides))
    initial = read_validated()

    def staged_value() -> str:
        current = read_validated()
        return original if current == initial else saf_overrides_json(current)

    _add_restore_button(qt, layout, zh, "saf_restore_defaults", lambda: show_values({}))
    return staged_value


# ---------------------------------------------------------------------------
# Staged Qt panels — Audited other algorithm families
# ---------------------------------------------------------------------------

def add_staged_denoise_fields(layout: Any, node: Any, zh: bool):
    """Build staged DWI denoising settings (DIPY Gibbs ringing removal toggle)."""

    import qt

    gibbs_box = qt.QCheckBox()
    gibbs_box.setObjectName("denoise_dipy_run_gibbs")
    gibbs_box.setToolTip(
        "在 MP-PCA 去噪后执行 DIPY 吉布斯振铃去除；部分傅里叶采集数据可取消勾选。"
        if zh
        else "Run DIPY Gibbs ringing removal after MP-PCA denoising; uncheck for partial-Fourier acquisitions."
    )
    layout.addRow("去除吉布斯振铃" if zh else "Gibbs ringing removal", gibbs_box)

    original = get_node_bool_parameter(node, "preprocessingRunGibbs", True)
    gibbs_box.setChecked(original)
    initial = _read_check(gibbs_box)

    def staged_value() -> bool:
        current = _read_check(gibbs_box)
        return original if current == initial else current

    _add_restore_button(qt, layout, zh, "denoise_restore_defaults", lambda: gibbs_box.setChecked(True))
    return staged_value


MICRO_DTI_FIT_METHOD_ITEMS = (
    ("wls", "Weighted least squares (WLS)", "加权最小二乘 (WLS)"),
    ("ls", "Ordinary least squares (LS)", "普通最小二乘 (LS)"),
)


def add_staged_microstructure_dti_fields(layout: Any, node: Any, zh: bool, *, dark: Any = None):
    """Build staged Microstructure DTI tensor fit method control."""

    import qt

    combo, set_code, get_code = _make_combo(
        qt, MICRO_DTI_FIT_METHOD_ITEMS, zh, "micro_dti_fit_method", dark=dark,
    )
    combo.setToolTip(
        "微结构 DTI 张量拟合方法（默认 WLS 加权最小二乘）。"
        if zh
        else "Tensor fitting estimator for microstructure DTI maps (default: weighted least squares)."
    )
    layout.addRow("张量拟合方法" if zh else "Tensor fit method", combo)

    raw_orig = str(get_node_parameter(node, "microstructureFitMethod", "wls") or "wls").strip().lower()
    original = validate_microstructure_dti_fit_method(raw_orig if raw_orig in {"wls", "ls", "ols"} else "wls")
    set_code(original)
    initial = get_code()

    def staged_value() -> str:
        current = validate_microstructure_dti_fit_method(get_code())
        return original if current == initial else current

    _add_restore_button(qt, layout, zh, "micro_dti_restore_defaults", lambda: set_code("wls"))
    return staged_value


def add_staged_freewater_fields(layout: Any, node: Any, zh: bool):
    """Build staged Free Water parameter controls."""

    import qt

    max_fw_spin = qt.QDoubleSpinBox()
    max_fw_spin.setRange(0.05, 0.99)
    max_fw_spin.setSingleStep(0.05)
    max_fw_spin.setDecimals(2)
    max_fw_spin.setObjectName("freewater_max_fraction_for_tissue")
    max_fw_spin.setToolTip(
        "计算校正组织 FA/MD 时允许的最大自由水比例（DIPY 参考默认 0.70）。"
        if zh
        else "Maximum fitted free-water fraction retained for corrected tissue FA/MD (DIPY reference default: 0.70)."
    )
    layout.addRow(
        "组织指标最大自由水比例" if zh else "Max FW fraction for tissue",
        max_fw_spin,
    )

    def show_values(raw_values: Mapping[str, Any]) -> None:
        overrides = parse_freewater_overrides(raw_values)
        max_fw_spin.setValue(float(overrides.get("max_fraction_for_tissue", 0.7)))

    def read_dict() -> dict[str, Any]:
        val = _read_spin(max_fw_spin, decimals=2)
        return parse_freewater_overrides({"max_fraction_for_tissue": val})

    original = str(get_node_parameter(node, "freeWaterParameters", "{}") or "{}").strip() or "{}"
    show_values(_safe_load_overrides(original, parse_freewater_overrides))
    initial = read_dict()

    def staged_value() -> str:
        current = read_dict()
        return original if current == initial else _minimal_overrides_json(current, FREEWATER_DEFAULTS)

    _add_restore_button(qt, layout, zh, "freewater_restore_defaults", lambda: show_values({}))
    return staged_value


def add_staged_ifw_t2_fields(layout: Any, node: Any, zh: bool):
    """Build staged iFW researcher T2 relaxation time controls (Cho et al. defaults: 95 / 1500 ms)."""

    import qt

    gm_spin = qt.QDoubleSpinBox()
    gm_spin.setRange(1.0, 5000.0)
    gm_spin.setSingleStep(5.0)
    gm_spin.setDecimals(1)
    gm_spin.setObjectName("ifw_t2_gm_ms")
    gm_spin.setToolTip(
        "灰质/组织 T2 弛豫时间（毫秒；Cho et al. 默认 95 ms）。"
        if zh
        else "Tissue T2 relaxation time in ms (Cho et al. reference default: 95 ms)."
    )
    layout.addRow("组织 T2 (ms)" if zh else "Tissue T2 (ms)", gm_spin)

    csf_spin = qt.QDoubleSpinBox()
    csf_spin.setRange(1.0, 5000.0)
    csf_spin.setSingleStep(50.0)
    csf_spin.setDecimals(1)
    csf_spin.setObjectName("ifw_t2_csf_ms")
    csf_spin.setToolTip(
        "脑脊液 T2 弛豫时间（毫秒；Cho et al. 默认 1500 ms）。"
        if zh
        else "CSF T2 relaxation time in ms (Cho et al. reference default: 1500 ms)."
    )
    layout.addRow("脑脊液 T2 (ms)" if zh else "CSF T2 (ms)", csf_spin)

    def show_values(raw_values: Mapping[str, Any]) -> None:
        overrides = parse_ifw_overrides(raw_values)
        gm_spin.setValue(float(overrides.get("t2_gm_ms", 95.0)))
        csf_spin.setValue(float(overrides.get("t2_csf_ms", 1500.0)))

    def read_dict() -> dict[str, Any]:
        return parse_ifw_overrides({
            "t2_gm_ms": _read_spin(gm_spin, decimals=1),
            "t2_csf_ms": _read_spin(csf_spin, decimals=1),
        })

    original = str(get_node_parameter(node, "ifwParameters", "{}") or "{}").strip() or "{}"
    show_values(_safe_load_overrides(original, parse_ifw_overrides))
    initial = read_dict()

    def staged_value() -> str:
        current = read_dict()
        return original if current == initial else _minimal_overrides_json(current, IFW_T2_DEFAULTS)

    _add_restore_button(qt, layout, zh, "ifw_restore_defaults", lambda: show_values({}))
    return staged_value


def add_staged_tractcloud_fields(layout: Any, node: Any, zh: bool):
    """Build staged TractCloud inference batch-size control."""

    import qt

    batch_spin = qt.QSpinBox()
    batch_spin.setRange(64, 16384)
    batch_spin.setSingleStep(128)
    batch_spin.setObjectName("tractcloud_batch_size")
    batch_spin.setToolTip(
        "TractCloud 推理批大小（默认 2048；显存不足时会自动下调上限）。"
        if zh
        else "TractCloud inference batch size (default: 2048; automatically capped on low-VRAM GPUs)."
    )
    layout.addRow("推理批大小" if zh else "Inference batch size", batch_spin)

    def show_values(raw_values: Mapping[str, Any]) -> None:
        overrides = parse_tractcloud_overrides(raw_values)
        batch_spin.setValue(int(overrides.get("batch_size", 2048)))

    def read_dict() -> dict[str, Any]:
        return parse_tractcloud_overrides({"batch_size": _read_spin(batch_spin, as_int=True)})

    original = str(get_node_parameter(node, "tractcloudParameters", "{}") or "{}").strip() or "{}"
    show_values(_safe_load_overrides(original, parse_tractcloud_overrides))
    initial = read_dict()

    def staged_value() -> str:
        current = read_dict()
        return original if current == initial else _minimal_overrides_json(current, TRACTCLOUD_DEFAULTS)

    _add_restore_button(qt, layout, zh, "tractcloud_restore_defaults", lambda: show_values({}))
    return staged_value
