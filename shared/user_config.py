"""Cross-platform user configuration shared by the Slicer UI and CLI."""

from __future__ import annotations

import json
import os
import platform
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from shared.dependency_config import DEPENDENCY_CONFIG_KEYS
from shared.branding import LEGACY_ENV_PREFIX, reuse_legacy_path


USER_CONFIG_KIND = "slicerdsurfer_user_config"
USER_CONFIG_VERSION = 1
USER_CONFIG_ENV = "SLICERDSPACE_USER_CONFIG"

RUN_DEFAULTS = {
    "run_name": "Test1",
    "targets": ["microstructure", "tractography", "bundles", "connectome"],
    "preprocessing_backend": "dipy",
    "brain_mask_backend": "pnl_cnn",
    "brain_mask_device": "auto",
    "ddseg_device": "auto",
    "surface_device": "auto",
    "tractcloud_device": "auto",
    "tractography_backend": "csd_act",
    "bundle_backend": "wmql",
    "wma_atlas": "org",
    "microstructure_stages": ["dti"],
    "dki_fit_method": "MKCURVE",
    "connectome_backend": "auto",
    "select_streamlines": 1_000_000,
    "auto_install_python_dependencies": True,
}


MICROSTRUCTURE_STAGE_ORDER = ("dti", "freewater", "noddi", "dki")


def canonical_microstructure_stages(values: Iterable[str] | str | None) -> tuple[str, ...]:
    """Keep DTI enabled while ordering independently selected optional stages."""

    if isinstance(values, str):
        values = [values]
    stages = {
        token.strip().lower()
        for value in values or ()
        for token in str(value).replace("+", ",").split(",")
        if token.strip()
    }
    if stages & {"all", "full"}:
        stages.update(MICROSTRUCTURE_STAGE_ORDER)
        stages.difference_update(("all", "full"))
    unknown = stages - set(MICROSTRUCTURE_STAGE_ORDER)
    if unknown:
        raise ValueError("Unknown microstructure stages: " + ", ".join(sorted(unknown)))
    stages.add("dti")
    return tuple(stage for stage in MICROSTRUCTURE_STAGE_ORDER if stage in stages)


def default_user_config_path(
    *,
    system: str | None = None,
    home: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Return the user-local JSON configuration path for one platform."""

    env = dict(os.environ if environ is None else environ)
    explicit = str(env.get(USER_CONFIG_ENV) or env.get(LEGACY_ENV_PREFIX + 'USER_CONFIG') or "").strip()
    if explicit:
        return Path(explicit).expanduser()
    resolved_home = (home or Path.home()).expanduser()
    system_name = (system or platform.system()).lower()
    if system_name.startswith("win"):
        base = Path(env.get("APPDATA") or env.get("LOCALAPPDATA") or resolved_home / "AppData" / "Roaming")
    elif system_name == "darwin":
        base = resolved_home / "Library" / "Application Support"
    else:
        base = Path(env.get("XDG_CONFIG_HOME") or resolved_home / ".config")
    return reuse_legacy_path(base / "SlicerDSpace" / "user_config.json")


def empty_user_config() -> dict[str, Any]:
    """Return a new valid configuration payload."""

    return {
        "kind": USER_CONFIG_KIND,
        "version": USER_CONFIG_VERSION,
        "updated_at": None,
        "ui": {
            "language": "en",
            "first_run_wizard_shown": False,
        },
        "recent": {
            "subject_id": "",
            "subject_root": "",
            "output_root": "",
        },
        "run_defaults": deepcopy(RUN_DEFAULTS),
        "external_dependencies": {},
    }


def _merge_dict(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def normalize_user_config(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize a user configuration while preserving forward-compatible keys."""

    config = empty_user_config()
    if payload:
        _merge_dict(config, payload)
    config["kind"] = USER_CONFIG_KIND
    config["version"] = USER_CONFIG_VERSION
    for section, fallback in (
        ("ui", empty_user_config()["ui"]),
        ("recent", empty_user_config()["recent"]),
        ("run_defaults", empty_user_config()["run_defaults"]),
    ):
        if not isinstance(config.get(section), Mapping):
            config[section] = deepcopy(fallback)
    language = str(config["ui"].get("language") or "en").lower()
    config["ui"]["language"] = "zh" if language.startswith("zh") else "en"
    dependencies = config.get("external_dependencies")
    config["external_dependencies"] = {
        key: str(value).strip()
        for key, value in (dependencies.items() if isinstance(dependencies, Mapping) else ())
        if key in DEPENDENCY_CONFIG_KEYS and str(value or "").strip()
    }
    defaults = config.get("run_defaults", {})
    targets = defaults.get("targets")
    if isinstance(targets, str):
        defaults["targets"] = [token.strip() for token in targets.split(",") if token.strip()]
    stages = defaults.get("microstructure_stages")
    defaults["microstructure_stages"] = list(canonical_microstructure_stages(stages))
    defaults["dki_fit_method"] = str(defaults.get("dki_fit_method") or "MKCURVE").strip().upper()
    return config


def load_user_config(path: Path | None = None, *, missing_ok: bool = True) -> dict[str, Any]:
    """Load and normalize the shared user JSON configuration."""

    resolved = (path or default_user_config_path()).expanduser()
    if not resolved.is_file():
        if missing_ok:
            return empty_user_config()
        raise FileNotFoundError(f"SlicerDSpace user config not found: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"SlicerDSpace user config must contain a JSON object: {resolved}")
    if payload.get("kind") not in {None, USER_CONFIG_KIND}:
        raise ValueError(f"Unsupported SlicerDSpace user config kind: {payload.get('kind')}")
    return normalize_user_config(payload)


def write_user_config(payload: Mapping[str, Any], path: Path | None = None) -> Path:
    """Atomically write the shared user configuration JSON."""

    resolved = (path or default_user_config_path()).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    config = normalize_user_config(payload)
    config["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    temporary = resolved.with_name(resolved.name + ".tmp")
    temporary.write_text(json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, resolved)
    return resolved


def update_user_config(
    updates: Mapping[str, Any],
    path: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    """Merge updates into the existing configuration and write it."""

    config = load_user_config(path)
    _merge_dict(config, updates)
    written = write_user_config(config, path)
    return normalize_user_config(config), written


def user_config_cli_values(config: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten reusable user settings into argparse destination names."""

    normalized = normalize_user_config(config)
    recent = normalized["recent"]
    defaults = normalized["run_defaults"]
    values: dict[str, Any] = {
        "subject_id": recent.get("subject_id") or None,
        "subject_root": Path(recent["subject_root"]).expanduser() if recent.get("subject_root") else None,
        "output_root": Path(recent["output_root"]).expanduser() if recent.get("output_root") else None,
        "run_name": defaults.get("run_name") or RUN_DEFAULTS["run_name"],
        "targets": [",".join(defaults.get("targets") or RUN_DEFAULTS["targets"])],
        "preprocessing_backend": defaults.get("preprocessing_backend") or RUN_DEFAULTS["preprocessing_backend"],
        "brain_mask_backend": defaults.get("brain_mask_backend") or RUN_DEFAULTS["brain_mask_backend"],
        "brain_mask_device": defaults.get("brain_mask_device") or RUN_DEFAULTS["brain_mask_device"],
        "ddseg_device": defaults.get("ddseg_device") or RUN_DEFAULTS["ddseg_device"],
        "surface_device": defaults.get("surface_device") or RUN_DEFAULTS["surface_device"],
        "tractcloud_device": defaults.get("tractcloud_device") or RUN_DEFAULTS["tractcloud_device"],
        "tractography_backend": defaults.get("tractography_backend") or RUN_DEFAULTS["tractography_backend"],
        "bundle_backend": defaults.get("bundle_backend") or RUN_DEFAULTS["bundle_backend"],
        "wma_atlas": defaults.get("wma_atlas") or "org",
        "microstructure_stages": [",".join(defaults.get("microstructure_stages") or RUN_DEFAULTS["microstructure_stages"])],
        "dki_fit_method": defaults.get("dki_fit_method") or "MKCURVE",
        "connectome_backend": defaults.get("connectome_backend") or RUN_DEFAULTS["connectome_backend"],
        "select_streamlines": int(defaults.get("select_streamlines") or RUN_DEFAULTS["select_streamlines"]),
    }
    for key, value in normalized["external_dependencies"].items():
        values[key] = value if key == "python_environment" else Path(value).expanduser()
    return values


def apply_user_config_to_args(args: Any, config: Mapping[str, Any], *, explicit_dests: set[str]) -> None:
    """Apply saved values only when the corresponding CLI option was not explicit."""

    for key, value in user_config_cli_values(config).items():
        if key in explicit_dests or value is None:
            continue
        setattr(args, key, value)


def explicit_cli_destinations(parser: Any, argv: list[str]) -> set[str]:
    """Return argparse destinations explicitly present in argv."""

    option_dest = {
        option: action.dest
        for action in parser._actions
        for option in getattr(action, "option_strings", ())
    }
    explicit: set[str] = set()
    for token in argv:
        option = token.split("=", 1)[0]
        destination = option_dest.get(option)
        if destination:
            explicit.add(destination)
    return explicit


def coupled_explicit_cli_destinations(explicit_dests: set[str]) -> set[str]:
    """Prevent saved subject metadata from leaking into an explicit new subject root."""

    coupled = set(explicit_dests)
    if "subject_root" in explicit_dests:
        coupled.update(("subject_id", "output_root"))
    return coupled


def user_config_from_cli_args(
    args: Any,
    *,
    language: str = "en",
    base_config: Mapping[str, Any] | None = None,
    resolved_targets: Iterable[str] | None = None,
    resolved_stages: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build a reusable configuration payload from resolved CLI arguments."""

    targets = [str(value).strip() for value in (resolved_targets or ()) if str(value).strip()]
    if not targets:
        for value in getattr(args, "targets", None) or []:
            targets.extend(token.strip() for token in str(value).split(",") if token.strip())
    stages = [str(value).strip() for value in (resolved_stages or ()) if str(value).strip()]
    if not stages:
        for value in getattr(args, "microstructure_stages", None) or []:
            stages.extend(token.strip() for token in str(value).split(",") if token.strip())
    target_set = set(targets)
    stage_set = set(stages)
    if "all" in stage_set:
        stage_set.update(MICROSTRUCTURE_STAGE_ORDER)
    if "microstructure_freewater" in target_set:
        stage_set.update(("dti", "freewater"))
    if "microstructure_noddi" in target_set:
        stage_set.update(("dti", "noddi"))
    if "microstructure_dki" in target_set or bool(getattr(args, "run_dki", False)):
        stage_set.update(("dti", "dki"))
    stages = list(canonical_microstructure_stages(stage_set))
    tractography_backend = getattr(args, "tractography_backend", None) or "csd_act"
    if "saf" in target_set:
        tractography_backend = "saf"
    bundle_backend = getattr(args, "bundle_backend", None) or "wmql"
    selected_bundle_targets = target_set & {"bundles", "wma", "tractcloud"}
    if selected_bundle_targets == {"wma"}:
        bundle_backend = "wma"
    elif selected_bundle_targets == {"tractcloud"}:
        bundle_backend = "tractcloud"
    elif selected_bundle_targets == {"bundles", "wma", "tractcloud"}:
        bundle_backend = "all"
    external = {
        key: Path(value).expanduser().as_posix() if key != "python_environment" else str(value)
        for key in DEPENDENCY_CONFIG_KEYS
        if (value := getattr(args, key, None)) is not None
    }
    subject_root = getattr(args, "subject_root", None)
    output_root = getattr(args, "output_root", None)
    config = normalize_user_config(base_config)
    config["ui"]["language"] = language
    config["recent"].update(
        {
            "subject_id": str(getattr(args, "subject_id", "") or ""),
            "subject_root": Path(subject_root).expanduser().as_posix() if subject_root else "",
            "output_root": Path(output_root).expanduser().as_posix() if output_root else "",
        }
    )
    config["run_defaults"].update(
        {
            "run_name": str(getattr(args, "run_name", "") or "Test1"),
            "targets": targets or list(RUN_DEFAULTS["targets"]),
            "preprocessing_backend": getattr(args, "preprocessing_backend", None) or "dipy",
            "brain_mask_backend": getattr(args, "brain_mask_backend", None) or "pnl_cnn",
            "brain_mask_device": getattr(args, "brain_mask_device", None) or "auto",
            "ddseg_device": getattr(args, "ddseg_device", None) or "auto",
            "surface_device": getattr(args, "surface_device", None) or "auto",
            "tractcloud_device": getattr(args, "tractcloud_device", None) or "auto",
            "tractography_backend": tractography_backend,
            "bundle_backend": bundle_backend,
            "wma_atlas": getattr(args, "wma_atlas", None) or "org",
            "microstructure_stages": stages or ["dti"],
            "dki_fit_method": str(getattr(args, "dki_fit_method", None) or "MKCURVE").upper(),
            "connectome_backend": getattr(args, "connectome_backend", None) or "auto",
            "select_streamlines": int(
                getattr(args, "select_streamlines", None) or RUN_DEFAULTS["select_streamlines"]
            ),
        }
    )
    config["external_dependencies"].update(external)
    return normalize_user_config(config)
