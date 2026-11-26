"""Hot on-demand model/atlas detection, network/offline controls, and in-app supplement prompt."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable

from shared.model_assets import (
    bundled_model_asset_roots,
    configured_model_assets_root,
)
from shared.network_config import (
    PRESET_MIRRORS,
    load_network_settings,
    load_release_packs_manifest,
    save_network_settings,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

_DEFAULT_PACK_SIZES_BYTES = {
    "pnl_cnn": 233_000_000,
    "segmentation": 360_000_000,
    "parcellation": 410_000_000,
    "surface": 255_000_000,
    "surface_postprocess": 70_000_000,
    "tractcloud": 35_000_000,
    "wma_org": 1_500_000_000,
    "wma_naba": 2_900_000_000,
    "wma_east_west": 9_900_000_000,
}

_RESOURCE_LABELS = {
    "pnl_cnn": ("PNL CNN 自动脑掩膜权重", "PNL CNN Brain-Mask Weights"),
    "segmentation": ("DDSeg 组织分割模型权重", "DDSeg Tissue-Segmentation Weights"),
    "parcellation": ("DDParcel 101脑区分区权重与图谱模板", "DDParcel 101-ROI Parcellation Weights & Template"),
    "surface": ("DDSurfer 皮层表面重建模型权重", "DDSurfer Cortical Surface Weights"),
    "surface_postprocess": ("FreeSurfer 皮层表面图谱与原生后处理组件", "FreeSurfer Surface Atlas & Runtime"),
    "tractcloud": ("TractCloud 白质束深度学习权重", "TractCloud Deep-Learning Bundle Weights"),
    "wma_org": ("WMA ORG 白质纤维图谱", "WMA ORG White-Matter Atlas"),
    "wma_naba": ("WMA NABA 全龄白质图谱", "WMA NABA White-Matter Atlas"),
    "wma_east_west": ("WMA East-West 白质图谱", "WMA East-West White-Matter Atlas"),
}

# The optional-resource profiles of scripts/setup_optional_assets.py -> the resource ids above.
_PROFILE_RESOURCE_IDS = {
    "wma": "wma_org",
    "wma-naba": "wma_naba",
    "wma-east-west": "wma_east_west",
    "tractcloud": "tractcloud",
    "pnl-cnn": "pnl_cnn",
    "surface-postprocess": "surface_postprocess",
}


def optional_profile_title(profile: str, zh: bool) -> str:
    """The friendly title of one optional-resource profile (b2-setup); an unknown profile keeps its id."""
    labels = _RESOURCE_LABELS.get(_PROFILE_RESOURCE_IDS.get(str(profile), ""))
    if not labels:
        return str(profile)
    return labels[0] if zh else labels[1]


def core_profile_title(profile: str, zh: bool) -> str:
    """The friendly title of one core model-asset profile (surface / segmentation / parcellation)."""
    labels = _RESOURCE_LABELS.get(str(profile))
    if not labels:
        return str(profile)
    return labels[0] if zh else labels[1]


def _text(zh: bool, english: str, chinese: str) -> str:
    return chinese if zh else english


def _format_size(num_bytes: int) -> str:
    size = float(max(0, int(num_bytes)))
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _candidate_asset_roots(explicit_root: Path | str | None = None) -> tuple[Path, ...]:
    # The bundled root is read with a configured root too: setup saves model_assets_dir after the first on-demand
    # install, and the base package still ships DDSeg and DDParcel in bundled-model-assets.
    roots = [configured_model_assets_root(explicit_root), *bundled_model_asset_roots(REPO_ROOT)]
    deduped: list[Path] = []
    for root in roots:
        try:
            resolved = root.expanduser().resolve()
        except OSError:
            resolved = root.expanduser()
        if resolved not in deduped:
            deduped.append(resolved)
    return tuple(deduped)


def _all_files_nonempty(roots: Iterable[Path], relatives: Iterable[str]) -> bool:
    rel_tuple = tuple(relatives)
    for root in roots:
        if all((root / rel).is_file() and (root / rel).stat().st_size > 0 for rel in rel_tuple):
            return True
    return False


def _json_manifest_sizes(filename: str) -> dict[str, int]:
    path = REPO_ROOT / "models" / filename
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get("asset_records") or {}
        return {
            rel: int(rec.get("size_bytes") or 0)
            for rel, rec in records.items()
            if isinstance(rec, dict)
        }
    except Exception:
        return {}


def _core_profile_paths(profile: str) -> tuple[str, ...]:
    try:
        payload = json.loads((REPO_ROOT / "models" / "assets_manifest.json").read_text(encoding="utf-8"))
        return tuple(payload["profiles"][profile]["required_paths"])
    except Exception:
        return ()


def _ext_profile_paths(profile: str) -> tuple[str, ...]:
    try:
        payload = json.loads((REPO_ROOT / "models" / "external_assets_manifest.json").read_text(encoding="utf-8"))
        spec = payload["profiles"][profile]
        paths = list(spec.get("required_paths") or ())
        for child in spec.get("include_profiles") or ():
            paths.extend(payload["profiles"][child].get("required_paths") or ())
        return tuple(dict.fromkeys(paths))
    except Exception:
        return ()


def _wma_atlas_ready(roots: tuple[Path, ...], atlas_family: str, explicit_wma_dir: str | None = None) -> bool:
    from tractography.wma_atlases import atlas_definition

    definition = atlas_definition(atlas_family)
    if explicit_wma_dir:
        explicit = Path(explicit_wma_dir).expanduser()
        if all((explicit / child).is_dir() for child in definition["directories"]):
            return True
    rel_dir = "bundles/wma" if atlas_family == "org" else f"bundles/wma/{atlas_family}"
    for root in roots:
        atlas_dir = root / rel_dir
        if (
            all((atlas_dir / child).is_dir() for child in definition["directories"])
            and (atlas_dir / "extracted_resource_index.json").is_file()
        ):
            return True
    return False


def _surface_runtime_state(explicit_fs_home: str | None = None, asset_root: str | None = None) -> dict[str, Any]:
    """b2-os: the structured state (ready / missing / os_blocked and the remedy) of the runtime the run would use.

    An explicit path that is a folder is kept, as the run keeps it (R3-3). It is ready only with a valid
    ``runtime.json`` or FreeSurfer's ``build-stamp.txt`` (decision 7), and never when it needs a newer macOS (G-H2).
    """
    from shared.surface_runtime import surface_runtime_state

    explicit = None
    if explicit_fs_home:
        candidate = Path(explicit_fs_home).expanduser()
        explicit = candidate if candidate.is_dir() else None
    try:
        return surface_runtime_state(freesurfer_home=explicit, asset_root=asset_root)
    except Exception:
        return {"state": "missing", "remedy": "none", "download_bytes": 0, "reason": "", "macos_required": "",
                "macos_current": ""}


def _surface_runtime_ready(explicit_fs_home: str | None = None) -> bool:
    return _surface_runtime_state(explicit_fs_home)["state"] == "ready"


def detect_missing_task_resources(
    owner: Any = None,
    *,
    targets: Iterable[str] | None = None,
    bundle_backends: Iterable[str] | None = None,
    wma_atlas: str | None = None,
    brain_mask_backend: str | None = None,
    pnl_mask_route: str | None = None,
    backend_paths: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Fast file-level check for model/atlas weights needed by currently active targets."""

    node = getattr(owner, "_parameterNode", None) if owner is not None else None
    if backend_paths is None:
        getter = getattr(owner, "_environment_dependency_path_kwargs", None)
        backend_paths = getter() if callable(getter) else {}
    backend_paths = dict(backend_paths or {})

    if targets is None:
        # b2-os F6: the closure after the OS skip, so the blocked Surface postprocess (and SAF) add no downloads;
        # prerequisites that a runnable target selects by itself stay.
        closure_fn = getattr(owner, "_os_filtered_closure", None) or getattr(owner, "_selected_closure", None)
        closure = closure_fn() if callable(closure_fn) else None
        if closure is not None:
            active_targets = tuple(closure.targets)
        else:
            from shared.target_closure import expand_closure

            active_targets = expand_closure(()).targets
    else:
        active_targets = tuple(targets)

    target_set = set(active_targets)
    if bundle_backends is None:
        from shared.algorithm_selection import BUNDLES, algorithms

        bundle_text = str(getattr(node, "orchestrationBundleBackend", "") or "wmql") if node else "wmql"
        bundle_tuple = algorithms(bundle_text, BUNDLES, "wmql")
    else:
        bundle_tuple = tuple(bundle_backends)

    wma_family = str(wma_atlas or (getattr(node, "wmaAtlas", "org") if node else "org") or "org").lower()
    mask_backend = str(brain_mask_backend or (getattr(node, "brainMaskBackend", "pnl_cnn") if node else "pnl_cnn") or "pnl_cnn")
    pnl_route = str(pnl_mask_route or (getattr(node, "pnlMaskRoute", "auto") if node else "auto") or "auto")

    explicit_root = backend_paths.get("model_assets_dir")
    roots = _candidate_asset_roots(explicit_root)
    core_sizes = _json_manifest_sizes("assets_manifest.json")
    ext_sizes = _json_manifest_sizes("external_assets_manifest.json")
    packs = (load_release_packs_manifest().get("packs") or {})

    missing: list[dict[str, Any]] = []

    def add_item(item_id: str, size_bytes: int, pack_id: str | None = None, reason: str | None = None) -> None:
        zh_title, en_title = _RESOURCE_LABELS[item_id]
        item = {
            "id": item_id,
            "title_zh": zh_title,
            "title_en": en_title,
            "size_bytes": int(size_bytes),
            "pack_id": pack_id,
        }
        if reason:
            item["reason"] = reason
        missing.append(item)

    # 1. Anatomical priors (DDSeg / DDParcel / DDSurfer / Surface Postprocess)
    if "segmentation" in target_set:
        paths = _core_profile_paths("segmentation")
        if paths and not _all_files_nonempty(roots, paths):
            pack_size = int((packs.get("ddseg") or {}).get("size_bytes") or 0)
            size = pack_size or sum(core_sizes.get(p, 0) for p in paths) or _DEFAULT_PACK_SIZES_BYTES["segmentation"]
            add_item("segmentation", size, "ddseg")

    if "parcellation" in target_set:
        paths = _core_profile_paths("parcellation")
        if paths and not _all_files_nonempty(roots, paths):
            pack_size = int((packs.get("ddparcel") or {}).get("size_bytes") or 0)
            size = pack_size or sum(core_sizes.get(p, 0) for p in paths) or _DEFAULT_PACK_SIZES_BYTES["parcellation"]
            add_item("parcellation", size, "ddparcel")

    if "surface" in target_set:
        paths = _core_profile_paths("surface")
        if paths and not _all_files_nonempty(roots, paths):
            pack_size = int((packs.get("ddsurfer") or {}).get("size_bytes") or 0)
            size = pack_size or sum(core_sizes.get(p, 0) for p in paths) or _DEFAULT_PACK_SIZES_BYTES["surface"]
            add_item("surface", size, "ddsurfer")

    if "surface_postprocess" in target_set:
        atlas_paths = _ext_profile_paths("surface-postprocess")
        atlas_ok = bool(atlas_paths) and _all_files_nonempty(roots, atlas_paths)
        fs_candidate = (
            backend_paths.get("postprocess_freesurfer_home")
            or backend_paths.get("postprocess_runtime")
            or backend_paths.get("freesurfer_home")
        )
        runtime = _surface_runtime_state(fs_candidate, explicit_root)
        runtime_ok = runtime["state"] == "ready"
        # b2-os, decision 2: a runtime that a download cannot fix (remedy 'none', for example a Mac below the catalog's
        # min_macos) gives no item: no size, no "Download and Install". The environment check and the Run skip say why.
        if runtime_ok or runtime["remedy"] == "download":
            if not atlas_ok or not runtime_ok:
                size = (sum(ext_sizes.get(p, 0) for p in atlas_paths) if not atlas_ok else 0) + (
                    int(runtime["download_bytes"]) if not runtime_ok else 0
                )
                reason = runtime.get("reason") or None if not runtime_ok else None
                add_item("surface_postprocess", size or _DEFAULT_PACK_SIZES_BYTES["surface_postprocess"], "ddsurfer",
                         reason)

    # 2. Automatic PNL brain mask
    needs_mask = bool(
        target_set & {"preprocessing", "microstructure", "segmentation", "parcellation", "surface", "tractography", "bundles", "connectome", "saf"}
    )
    if needs_mask and mask_backend == "pnl_cnn":
        from shared.brain_mask import resolve_pnl_mask_route

        if resolve_pnl_mask_route(pnl_route) == "native":
            explicit_pnl = backend_paths.get("pnl_mask_model_dir")
            pnl_paths = _ext_profile_paths("pnl-cnn")
            pnl_ok = False
            if explicit_pnl:
                pnl_dir = Path(explicit_pnl).expanduser()
                if all((pnl_dir / Path(p).name).is_file() for p in pnl_paths):
                    pnl_ok = True
            if not pnl_ok and pnl_paths and _all_files_nonempty(roots, pnl_paths):
                pnl_ok = True
            if not pnl_ok:
                pnl_dir = Path(explicit_pnl).expanduser() if explicit_pnl else None
                truly_missing = [
                    p
                    for p in pnl_paths
                    if not (pnl_dir is not None and (pnl_dir / Path(p).name).is_file())
                    and not any((r / p).is_file() and (r / p).stat().st_size > 0 for r in roots)
                ]
                iit_rel = "brain_mask/pnl_cnn_torch/IITmean_b0_256.nii.gz"
                if truly_missing == [iit_rel]:
                    size = ext_sizes.get(iit_rel, 7_830_929)
                    missing.append(
                        {
                            "id": "pnl_cnn",
                            "title_zh": "PNL 脑掩膜参考模板（IITmean_b0_256.nii.gz）",
                            "title_en": "PNL Brain-Mask Template (IITmean_b0_256.nii.gz)",
                            "size_bytes": int(size),
                            "pack_id": "pnl_cnn",
                        }
                    )
                elif truly_missing and len(truly_missing) < len(pnl_paths):
                    size = sum(ext_sizes.get(p, 0) for p in truly_missing)
                    add_item("pnl_cnn", size or _DEFAULT_PACK_SIZES_BYTES["pnl_cnn"], "pnl_cnn")
                else:
                    pack_size = int((packs.get("pnl_cnn") or {}).get("size_bytes") or 0)
                    iit_size = int(ext_sizes.get(iit_rel, 7_830_929))
                    size = (pack_size + iit_size if pack_size else 0) or sum(ext_sizes.get(p, 0) for p in pnl_paths) or _DEFAULT_PACK_SIZES_BYTES["pnl_cnn"]
                    add_item("pnl_cnn", size, "pnl_cnn")

    # 3. Bundles: TractCloud / WMA. Fix round W3: the closure turns a selection without WMQL into the target 'wma' or
    # 'tractcloud' (not 'bundles'), and the bundle tuple can be empty or the 'wmql' default then; either form counts.
    wants_tractcloud = "tractcloud" in target_set or ("bundles" in target_set and "tractcloud" in bundle_tuple)
    wants_wma = "wma" in target_set or ("bundles" in target_set and "wma" in bundle_tuple)
    if wants_tractcloud or wants_wma:
        if wants_tractcloud:
            explicit_tc = backend_paths.get("tractcloud_data_dir")
            tc_ok = bool(explicit_tc and Path(explicit_tc).expanduser().is_dir() and (Path(explicit_tc).expanduser() / "TrainedModel/best_tract_f1_model.pth").is_file())
            if not tc_ok:
                tc_paths = _ext_profile_paths("tractcloud")
                tc_ok = bool(tc_paths) and _all_files_nonempty(roots, tc_paths)
            if not tc_ok:
                pack_size = int((packs.get("tractcloud") or {}).get("size_bytes") or 0)
                size = pack_size or sum(ext_sizes.get(p, 0) for p in _ext_profile_paths("tractcloud")) or _DEFAULT_PACK_SIZES_BYTES["tractcloud"]
                add_item("tractcloud", size, "tractcloud")

        if wants_wma:
            explicit_wma = backend_paths.get("wma_atlas_dir")
            if not _wma_atlas_ready(roots, wma_family, explicit_wma_dir=explicit_wma):
                if wma_family == "naba":
                    pack_size = int((packs.get("wma_naba") or {}).get("size_bytes") or 0)
                    add_item("wma_naba", pack_size or _DEFAULT_PACK_SIZES_BYTES["wma_naba"], "wma_naba")
                elif wma_family == "east-west":
                    pack_size = int((packs.get("wma_east_west") or {}).get("size_bytes") or 0)
                    add_item("wma_east_west", pack_size or _DEFAULT_PACK_SIZES_BYTES["wma_east_west"], "wma_east_west")
                else:
                    pack_size = int((packs.get("wma_org") or {}).get("size_bytes") or 0)
                    add_item("wma_org", pack_size or _DEFAULT_PACK_SIZES_BYTES["wma_org"], "wma_org")

    return missing


def _format_compact_size(num_bytes: int) -> str:
    b = int(num_bytes or 0)
    if b <= 0:
        return ""
    if b >= 1073741824:
        return f"~{b / 1073741824.0:.2f} GB"
    return f"~{max(1, int(round(b / 1048576.0)))} MB"


def installed_wma_atlases(
    owner: Any = None,
    *,
    backend_paths: dict[str, str] | None = None,
) -> tuple[str, ...]:
    """Return the WMA atlas families (subset of org, naba, east-west) currently present locally."""
    if backend_paths is None:
        getter = getattr(owner, "_environment_dependency_path_kwargs", None)
        backend_paths = getter() if callable(getter) else {}
    paths_map = dict(backend_paths or {})
    roots = _candidate_asset_roots(paths_map.get("model_assets_dir"))
    explicit_wma = paths_map.get("wma_atlas_dir")
    return tuple(
        family
        for family in ("org", "naba", "east-west")
        if _wma_atlas_ready(roots, family, explicit_wma_dir=explicit_wma)
    )


def detect_missing_direct_algorithms(
    owner: Any,
    family: str,
    *,
    zh: bool = False,
    backend_paths: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Fast check for direct missing model/atlas weights for one Analysis picker family.

    Only direct weight-bearing algorithms are checked (Anatomical priors: DDSurfer/DDSeg/DDParcel;
    Preprocessing: PNL CNN brain mask; Fiber bundles: TractCloud and WMA when no WMA atlas is installed).
    Downstream algorithms with indirect dependencies (CSD/ACT, SAF, WMQL, Connectome, Surface postprocess)
    are intentionally left enabled here and handled by the pre-run preparation prompt if needed.
    """
    if family not in ("anatomical_priors", "preprocessing", "bundles"):
        return {}
    if backend_paths is None:
        getter = getattr(owner, "_environment_dependency_path_kwargs", None)
        if not callable(getter):
            return {}
        try:
            backend_paths = getter() or {}
        except Exception:
            return {}
    paths_map = dict(backend_paths or {})
    node = getattr(owner, "_parameterNode", None) if owner is not None else None
    roots = _candidate_asset_roots(paths_map.get("model_assets_dir"))
    core_sizes = _json_manifest_sizes("assets_manifest.json")
    ext_sizes = _json_manifest_sizes("external_assets_manifest.json")
    packs = load_release_packs_manifest().get("packs") or {}
    out: dict[str, dict[str, Any]] = {}

    if family == "anatomical_priors":
        for field, profile, pack_id, en_name, zh_name in (
            ("targetSurface", "surface", "ddsurfer", "Cortical Surfaces", "皮层表面"),
            ("targetSegmentation", "segmentation", "ddseg", "Tissue Segmentation", "组织分割"),
            ("targetParcellation", "parcellation", "ddparcel", "Brain Parcellation", "大脑分区"),
        ):
            req_paths = _core_profile_paths(profile)
            if req_paths and not _all_files_nonempty(roots, req_paths):
                pack_size = int((packs.get(pack_id) or {}).get("size_bytes") or 0)
                size = pack_size or sum(core_sizes.get(p, 0) for p in req_paths) or _DEFAULT_PACK_SIZES_BYTES[profile]
                sz_txt = _format_compact_size(size)
                zh_title, en_title = _RESOURCE_LABELS[profile]
                out[field] = {
                    "id": profile,
                    "pack_id": pack_id,
                    "profile": profile,
                    "kind": "core",
                    "title_zh": zh_title,
                    "title_en": en_title,
                    "size_bytes": int(size),
                    "reason": (
                        f"未安装 {zh_title}（{sz_txt}）。补充下载后即可启用{zh_name}。"
                        if zh
                        else f"{en_title} ({sz_txt}) are not installed. Download them to enable {en_name}."
                    ),
                }
        return out

    if family == "preprocessing":
        mask_backend = str(getattr(node, "brainMaskBackend", "pnl_cnn") or "pnl_cnn") if node is not None else "pnl_cnn"
        pnl_route = str(getattr(node, "pnlMaskRoute", "auto") or "auto") if node is not None else "auto"
        if mask_backend == "pnl_cnn":
            from shared.brain_mask import resolve_pnl_mask_route

            if resolve_pnl_mask_route(pnl_route) == "native":
                explicit_pnl = paths_map.get("pnl_mask_model_dir")
                pnl_paths = _ext_profile_paths("pnl-cnn")
                pnl_ok = False
                if explicit_pnl:
                    pnl_dir = Path(explicit_pnl).expanduser()
                    if all((pnl_dir / Path(p).name).is_file() for p in pnl_paths):
                        pnl_ok = True
                if not pnl_ok and pnl_paths and _all_files_nonempty(roots, pnl_paths):
                    pnl_ok = True
                if not pnl_ok:
                    iit_rel = "brain_mask/pnl_cnn_torch/IITmean_b0_256.nii.gz"
                    pack_size = int((packs.get("pnl_cnn") or {}).get("size_bytes") or 0)
                    iit_size = int(ext_sizes.get(iit_rel, 7_830_929))
                    size = (pack_size + iit_size if pack_size else 0) or sum(ext_sizes.get(p, 0) for p in pnl_paths) or _DEFAULT_PACK_SIZES_BYTES["pnl_cnn"]
                    sz_txt = _format_compact_size(size)
                    zh_title, en_title = _RESOURCE_LABELS["pnl_cnn"]
                    out["brain_mask"] = {
                        "id": "pnl_cnn",
                        "pack_id": "pnl_cnn",
                        "profile": "pnl-cnn",
                        "kind": "optional",
                        "title_zh": zh_title,
                        "title_en": en_title,
                        "size_bytes": int(size),
                        "reason": (
                            f"未安装 {zh_title}（{sz_txt}）。补充下载后可自动提取脑掩膜；若输入目录已提供脑掩膜，则无需下载也可直接运行后续分析。"
                            if zh
                            else f"{en_title} ({sz_txt}) are not installed. Download them for automatic DWI brain masking, or provide your own brain mask under Inputs."
                        ),
                    }
        return out

    if family == "bundles":
        explicit_tc = paths_map.get("tractcloud_data_dir")
        tc_ok = bool(
            explicit_tc
            and Path(explicit_tc).expanduser().is_dir()
            and (Path(explicit_tc).expanduser() / "TrainedModel/best_tract_f1_model.pth").is_file()
        )
        if not tc_ok:
            tc_paths = _ext_profile_paths("tractcloud")
            tc_ok = bool(tc_paths) and _all_files_nonempty(roots, tc_paths)
        if not tc_ok:
            pack_size = int((packs.get("tractcloud") or {}).get("size_bytes") or 0)
            size = pack_size or sum(ext_sizes.get(p, 0) for p in _ext_profile_paths("tractcloud")) or _DEFAULT_PACK_SIZES_BYTES["tractcloud"]
            sz_txt = _format_compact_size(size)
            zh_title, en_title = _RESOURCE_LABELS["tractcloud"]
            out["tractcloud"] = {
                "id": "tractcloud",
                "pack_id": "tractcloud",
                "profile": "tractcloud",
                "kind": "optional",
                "title_zh": zh_title,
                "title_en": en_title,
                "size_bytes": int(size),
                "reason": (
                    f"未安装 {zh_title}（{sz_txt}）。补充下载后即可启用 TractCloud。"
                    if zh
                    else f"{en_title} ({sz_txt}) are not installed. Download them to enable TractCloud."
                ),
            }

        if not installed_wma_atlases(owner, backend_paths=paths_map):
            atlas_choices = []
            for a_fam, a_id, a_prof, a_name in (
                ("org", "wma_org", "wma", "ORG"),
                ("naba", "wma_naba", "wma-naba", "NABA"),
                ("east-west", "wma_east_west", "wma-east-west", "East-West"),
            ):
                sz = int((packs.get(a_id) or {}).get("size_bytes") or _DEFAULT_PACK_SIZES_BYTES[a_id])
                zh_t, en_t = _RESOURCE_LABELS[a_id]
                atlas_choices.append(
                    {
                        "atlas_family": a_fam,
                        "id": a_id,
                        "pack_id": a_id,
                        "profile": a_prof,
                        "kind": "optional",
                        "title_zh": zh_t,
                        "title_en": en_t,
                        "size_bytes": sz,
                        "label": f"{a_name} ({_format_compact_size(sz)})",
                    }
                )
            first = atlas_choices[0]
            out["wma"] = {
                **first,
                "atlas_choices": atlas_choices,
                "reason": (
                    "本地尚未安装任何 WMA 白质纤维图谱。请在下方选择所需图谱并下载，安装完成后即可启用 WMA。"
                    if zh
                    else "No WMA white-matter atlas is installed locally. Choose an atlas below to download and enable WMA."
                ),
            }
        return out

    return out


def build_single_resource_supplement_dialog(
    owner: Any,
    item_spec: dict[str, Any],
    *,
    zh: bool | None = None,
) -> tuple[Any, Callable[[], dict[str, str]]]:
    """Build the On-Demand Resource Supplement dialog for one missing model or atlas pack."""
    import qt
    import slicer

    use_zh = str(getattr(owner, "_onboardingLanguage", "en")).startswith("zh") if zh is None else bool(zh)
    item = {
        "id": str(item_spec.get("id") or ""),
        "title_zh": str(item_spec.get("title_zh") or item_spec.get("title_en") or item_spec.get("id") or ""),
        "title_en": str(item_spec.get("title_en") or item_spec.get("title_zh") or item_spec.get("id") or ""),
        "size_bytes": int(item_spec.get("size_bytes") or 0),
    }
    total_size = _format_size(item["size_bytes"])
    title_text = item["title_zh"] if use_zh else item["title_en"]
    headline = (
        f"即将下载并热补装 <b>{html.escape(title_text)}</b>（约 {total_size}），完成后即可在分析中勾选使用。"
        if use_zh
        else f"Download and install <b>{html.escape(title_text)}</b> (~{total_size}) to enable this algorithm?"
    )

    dialog = qt.QDialog(slicer.util.mainWindow() if hasattr(slicer.util, "mainWindow") else None)
    dialog.setObjectName("onDemandSupplementDialog")
    dialog.setProperty("popupCard", True)
    dialog.setWindowTitle(_text(use_zh, "On-Demand Resource Supplement", "按需补装模型与图谱"))
    try:
        from shared.ui_styles import apply_placeholder_palette, ui_stylesheet

        dialog.setStyleSheet(ui_stylesheet(widget=getattr(owner, "_uiWidget", None)))
        apply_placeholder_palette(dialog)
    except Exception:
        pass

    layout = qt.QVBoxLayout(dialog)
    layout.setContentsMargins(18, 16, 18, 16)
    layout.setSpacing(12)

    sub_note = (
        "无需重装扩展或重启 3D Slicer；可在下方选择加速镜像、代理或本地离线包目录。"
        if use_zh
        else "No extension reinstall or Slicer restart needed. Configure a mirror, proxy, or local offline pack folder below if needed."
    )
    if item["id"] == "pnl_cnn":
        sub_note += (
            " 若输入目录已提供脑掩膜，也可跳过下载直接运行后续分析。"
            if use_zh
            else " If you already provide a brain mask under Inputs, you may skip this download."
        )
    info = qt.QLabel(f"{headline}<br><br><span style='font-size:12px'>{html.escape(sub_note)}</span>", dialog)
    info.wordWrap = True
    layout.addWidget(info)

    net_panel, read_net_values = build_network_and_offline_panel(dialog, zh=use_zh)
    layout.addWidget(net_panel)

    buttons = qt.QDialogButtonBox(dialog)
    install_btn = buttons.addButton(
        _text(use_zh, "Download and Install", "一键下载补齐"),
        qt.QDialogButtonBox.AcceptRole,
    )
    buttons.addButton(
        _text(use_zh, "Cancel", "取消"),
        qt.QDialogButtonBox.RejectRole,
    )
    install_btn.setDefault(True)
    try:
        install_btn.setProperty("primaryAction", True)
        st = install_btn.style()
        if st is not None:
            st.unpolish(install_btn)
            st.polish(install_btn)
        install_btn.setFocus()
    except Exception:
        pass
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.setMinimumWidth(460)
    return dialog, read_net_values


def prompt_download_missing_resource(owner: Any, item_spec: dict[str, Any]) -> bool:
    """Confirm and asynchronously download one directly missing model/atlas pack from a picker '?' action."""
    if owner is None or not isinstance(item_spec, dict):
        return False
    import qt
    import slicer
    import sys

    if (
        getattr(owner, "_setupJob", None)
        or getattr(owner, "_pipelineActive", False)
        or getattr(owner, "_selectedSetupActive", False)
    ):
        zh = str(getattr(owner, "_onboardingLanguage", "en")).startswith("zh")
        slicer.util.infoDisplay(
            _text(zh, "Wait for the active job or cancel it first.", "请先等待当前任务结束，或取消当前任务。")
        )
        return False

    zh = str(getattr(owner, "_onboardingLanguage", "en")).startswith("zh")
    dialog, read_net_values = build_single_resource_supplement_dialog(owner, item_spec, zh=zh)
    accepted = dialog.exec_() == qt.QDialog.Accepted
    net_values = read_net_values()
    dialog.deleteLater()
    if not accepted:
        return False

    save_network_settings(**net_values)
    getter = getattr(owner, "_environment_dependency_path_kwargs", None)
    backend_paths = dict(getter() or {}) if callable(getter) else {}
    asset_root = configured_model_assets_root(backend_paths.get("model_assets_dir"))
    asset_root.mkdir(parents=True, exist_ok=True)

    launcher_fn = getattr(owner, "_slicer_launcher_program", None)
    launcher_prog = launcher_fn() if callable(launcher_fn) else None
    if launcher_prog:
        program, prefix = str(launcher_prog), ["--launch", "PythonSlicer"]
    else:
        program, prefix = sys.executable, []

    profile = str(item_spec.get("profile") or "")
    kind = str(item_spec.get("kind") or "optional")
    from datetime import datetime
    from shared.user_config import default_user_config_path

    log_folder = default_user_config_path().parent / "setup-history" / ("supplement-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    log_folder.mkdir(parents=True, exist_ok=True)

    if kind == "core":
        script_path = REPO_ROOT / "scripts" / "download_model_assets.py"
        args = [
            *prefix, "-u", str(script_path),
            "--profile", profile,
            "--asset-root", str(asset_root),
            "--source", "url",
            "--source-id", "slicerdsurfer_models_release",
        ]
    else:
        script_path = REPO_ROOT / "scripts" / "setup_optional_assets.py"
        receipt_path = log_folder / "optional-resources-result.json"
        args = [
            *prefix, "-u", str(script_path),
            "--profiles", profile,
            "--asset-root", str(asset_root),
            "--execute",
            "--accept-terms",
            "--result", str(receipt_path),
        ]

    title_text = str(item_spec.get("title_zh") if zh else item_spec.get("title_en") or profile)
    job_title = (f"正在下载 {title_text}" if zh else f"Downloading {title_text}")

    owner._selectedSetupActive = True
    try:
        from shared.ui_selected_setup import _safe_set_live_stage_text, _safe_set_run_state, _sync_owner_busy_controls

        _safe_set_run_state(owner, "setup")
        _safe_set_live_stage_text(owner, job_title + "…")
        _sync_owner_busy_controls(owner)
    except Exception:
        pass

    def _on_done(status: str, details: str, _log_path: Any) -> None:
        owner._selectedSetupActive = False
        owner._lastEnvironmentResult = None
        owner._lastEnvironmentSignature = None
        try:
            from shared.ui_selected_setup import _safe_set_run_state, _sync_owner_busy_controls

            _safe_set_run_state(owner, "idle" if status == "completed" else ("cancelled" if status == "cancelled" else "failed"))
            _sync_owner_busy_controls(owner)
        except Exception:
            pass
        if status == "completed":
            atlas_fam = item_spec.get("atlas_family")
            node = getattr(owner, "_parameterNode", None)
            if atlas_fam and node is not None:
                try:
                    with slicer.util.NodeModify(node):
                        node.wmaAtlas = str(atlas_fam)
                except Exception:
                    try:
                        node.wmaAtlas = str(atlas_fam)
                    except Exception:
                        pass
            try:
                from shared.ui_compact import refresh_compact_algorithms

                refresh_compact_algorithms(owner)
            except Exception:
                pass
            setter = getattr(owner, "_setStatus", None)
            if callable(setter):
                setter(
                    "ready",
                    f"已安装 {title_text}，现在可在 2. Analysis 中勾选使用。"
                    if zh
                    else f"Installed {title_text}. You can now select it under 2. Analysis.",
                )
        else:
            setter = getattr(owner, "_setStatus", None)
            if callable(setter):
                setter(
                    "warning",
                    f"下载未完成（{status}），请查看日志后重试。"
                    if zh
                    else f"Download did not finish ({status}); inspect logs and retry.",
                )

    from shared.ui_setup_jobs import start_setup_job

    start_setup_job(owner, job_title, [(program, args, {})], _on_done)
    return True


def build_network_and_offline_panel(parent: Any, *, zh: bool) -> tuple[Any, Callable[[], dict[str, str]]]:
    """Build a compact Mirror / Proxy / Offline Pack Directory panel for Qt setup dialogs."""

    import qt

    settings = load_network_settings()
    group = qt.QGroupBox(
        _text(zh, "Network acceleration, proxy and offline pack folder", "网络加速镜像 / 代理 / 本地离线包目录"),
        parent,
    )
    form = qt.QFormLayout(group)
    form.setContentsMargins(10, 8, 10, 8)
    form.setSpacing(6)

    mirror_combo = qt.QComboBox(group)
    for preset_id, prefix, en_label, zh_label in PRESET_MIRRORS:
        mirror_combo.addItem(_text(zh, en_label, zh_label), prefix)
    mirror_combo.addItem(_text(zh, "Custom mirror prefix...", "自定义镜像前缀..."), "__custom__")
    try:
        from shared.ui_styles import style_combobox_popup

        style_combobox_popup(mirror_combo)
    except Exception:
        pass

    custom_mirror = qt.QLineEdit(group)
    custom_mirror.setPlaceholderText("https://ghfast.top/")
    current_mirror = settings["mirror_prefix"]
    preset_index = next(
        (idx for idx, item in enumerate(PRESET_MIRRORS) if item[1] == current_mirror),
        None,
    )
    if preset_index is not None:
        mirror_combo.setCurrentIndex(preset_index)
        custom_mirror.setVisible(False)
    else:
        mirror_combo.setCurrentIndex(len(PRESET_MIRRORS))
        custom_mirror.setText(current_mirror)
        custom_mirror.setVisible(True)

    def _on_mirror_changed(index: int) -> None:
        data = str(mirror_combo.itemData(index) or "")
        custom_mirror.setVisible(data == "__custom__")

    mirror_combo.currentIndexChanged.connect(_on_mirror_changed)

    mirror_box = qt.QWidget(group)
    mirror_layout = qt.QVBoxLayout(mirror_box)
    mirror_layout.setContentsMargins(0, 0, 0, 0)
    mirror_layout.setSpacing(4)
    mirror_layout.addWidget(mirror_combo)
    mirror_layout.addWidget(custom_mirror)
    form.addRow(_text(zh, "Download source:", "下载镜像源："), mirror_box)

    proxy_edit = qt.QLineEdit(group)
    proxy_edit.setText(settings["proxy_url"])
    proxy_edit.setPlaceholderText(
        _text(zh, "Optional, e.g. http://127.0.0.1:7890", "可选，例如 http://127.0.0.1:7890")
    )
    form.addRow(_text(zh, "HTTP/HTTPS proxy:", "代理地址："), proxy_edit)

    offline_row = qt.QWidget(group)
    offline_layout = qt.QHBoxLayout(offline_row)
    offline_layout.setContentsMargins(0, 0, 0, 0)
    offline_layout.setSpacing(4)
    offline_edit = qt.QLineEdit(offline_row)
    offline_edit.setText(settings["offline_assets_dir"])
    offline_edit.setPlaceholderText(
        _text(zh, "Optional folder containing Pack-*.tar.gz / Pack-*.zip or model files", "可选：包含 Pack-*.tar.gz / Pack-*.zip 或离线模型文件的目录")
    )
    browse_btn = qt.QPushButton(_text(zh, "Browse...", "选择目录..."), offline_row)

    def _browse_offline() -> None:
        chosen = qt.QFileDialog.getExistingDirectory(
            group,
            _text(zh, "Select offline model pack directory", "选择本地离线模型/分包目录"),
            offline_edit.text.strip() or str(Path.home()),
        )
        if chosen:
            offline_edit.setText(chosen)

    browse_btn.clicked.connect(_browse_offline)
    offline_layout.addWidget(offline_edit, 1)
    offline_layout.addWidget(browse_btn)
    form.addRow(_text(zh, "Offline folder:", "本地离线目录："), offline_row)

    def get_values() -> dict[str, str]:
        data = str(mirror_combo.itemData(mirror_combo.currentIndex) or "")
        mirror = custom_mirror.text.strip() if data == "__custom__" else data
        return {
            "mirror_prefix": mirror,
            "proxy_url": proxy_edit.text.strip(),
            "offline_assets_dir": offline_edit.text.strip(),
        }

    return group, get_values


def _missing_bullets_html(missing: list[dict[str, Any]], zh: bool) -> str:
    """The bullet list of the supplement dialog; G-H2: an item with a ``reason`` (macOS too old) shows it."""
    return "".join(
        f"<br>• <b>{html.escape(item['title_zh'] if zh else item['title_en'])}</b> ({_format_size(item['size_bytes'])})"
        + (f"<br>&nbsp;&nbsp;<i>{html.escape(item['reason'])}</i>" if item.get("reason") else "")
        for item in missing
    )


def maybe_prompt_on_demand_install(owner: Any, *, force: bool = False) -> bool:
    """Prompt to hot-install missing model weights or atlases right when a user ticks a workflow."""

    if owner is None or getattr(owner, "_parameterNode", None) is None:
        return False
    if not force:
        if os.environ.get("SLICERDSPACE_DISABLE_AUTO_PROMPT") == "1":
            return False
        if (
            getattr(owner, "_updatingGuiFromParameterNode", False)
            or getattr(owner, "_isLoadingSettings", False)
            or getattr(owner, "_setupJob", None)
            or getattr(owner, "_pipelineActive", False)
            or getattr(owner, "_selectedSetupActive", False)
        ):
            return False
        try:
            import slicer

            if bool(getattr(slicer.app, "testingEnabled", lambda: False)()):
                return False
        except Exception:
            pass

    missing = detect_missing_task_resources(owner)
    if not missing:
        return False

    declined = set(getattr(owner, "_declinedOnDemandItems", set()) or ())
    new_ids = {item["id"] for item in missing}
    if not force and new_ids <= declined:
        return False

    import qt
    import slicer

    zh = str(getattr(owner, "_onboardingLanguage", "en")).startswith("zh")
    total_size = _format_size(sum(int(item["size_bytes"]) for item in missing))
    if len(missing) == 1:
        title_text = missing[0]["title_zh"] if zh else missing[0]["title_en"]
        headline = (
            f"检测到当前缺少 <b>{html.escape(title_text)}</b>（约 {total_size}），是否现在一键下载并热补装？"
            if zh
            else f"<b>{html.escape(title_text)}</b> (~{total_size}) is not installed locally. Download and install it now?"
        )
    else:
        headline = (
            f"所选分析缺少 {len(missing)} 项模型/图谱资源（合计约 {total_size}），是否一键补齐？"
            if zh
            else f"Selected analyses need {len(missing)} missing resource pack(s) (~{total_size}). Install them now?"
        )

    dialog = qt.QDialog(slicer.util.mainWindow())
    dialog.setObjectName("onDemandSupplementDialog")
    dialog.setProperty("popupCard", True)
    dialog.setWindowTitle(_text(zh, "On-Demand Resource Supplement", "按需补装模型与图谱"))
    try:
        from shared.ui_styles import apply_placeholder_palette, ui_stylesheet

        dialog.setStyleSheet(ui_stylesheet(widget=getattr(owner, "_uiWidget", None)))
        apply_placeholder_palette(dialog)
    except Exception:
        pass

    layout = qt.QVBoxLayout(dialog)
    layout.setContentsMargins(18, 16, 18, 16)
    layout.setSpacing(12)

    bullets = _missing_bullets_html(missing, zh)
    sub_note = (
        "无需重装扩展或重启 3D Slicer；可在下方选择加速镜像、代理或本地离线包目录。"
        if zh
        else "No extension reinstall or Slicer restart needed. Configure a mirror, proxy, or local offline pack folder below if needed."
    )
    if "pnl_cnn" in new_ids:
        extra_pnl = (
            " 若无法连接 NITRC 下载模板，也可在 Inputs 下提供已有脑掩膜（Provided mask）。"
            if zh
            else " If NITRC is unreachable, you may also supply an existing brain mask under Inputs (Provided mask)."
        )
        sub_note += extra_pnl
    info = qt.QLabel(f"{headline}{bullets}<br><br><span style='font-size:12px'>{html.escape(sub_note)}</span>", dialog)
    info.wordWrap = True
    layout.addWidget(info)

    net_panel, read_net_values = build_network_and_offline_panel(dialog, zh=zh)
    layout.addWidget(net_panel)

    buttons = qt.QDialogButtonBox(dialog)
    install_btn = buttons.addButton(
        _text(zh, "Download and Install", "一键下载补齐"),
        qt.QDialogButtonBox.AcceptRole,
    )
    later_btn = buttons.addButton(
        _text(zh, "Later", "稍后处理"),
        qt.QDialogButtonBox.RejectRole,
    )
    install_btn.setDefault(True)
    try:
        install_btn.setProperty("primaryAction", True)
    except Exception:
        pass
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.setMinimumWidth(460)

    accepted = dialog.exec_() == qt.QDialog.Accepted
    net_values = read_net_values()
    dialog.deleteLater()

    if not accepted:
        owner._declinedOnDemandItems = declined | new_ids
        return False

    # Clear declined flag for the items we are actively installing now.
    owner._declinedOnDemandItems = declined - new_ids
    save_network_settings(**net_values)
    owner._lastEnvironmentResult = None
    owner._lastEnvironmentSignature = None

    from shared.ui_environment_check import start_environment_check
    from shared.ui_selected_setup import prepare_selected_environment

    start_environment_check(
        owner,
        lambda report: prepare_selected_environment(owner, report, skip_confirm=True),
    )
    return True
