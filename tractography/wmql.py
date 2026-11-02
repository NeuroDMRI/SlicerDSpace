"""WMQL / Tract Querier integration for rule-based bundle extraction."""

from __future__ import annotations

from shared.cohort_schedule import scheduled_stage

import ast
import json
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional

from shared.external_process import python_tool_environment, text_subprocess_kwargs
from shared.output_contracts import polydata_streamline_count
from shared.product_staging import archive_run_records, new_product_staging, publish_products, retain_failed_products
from tractography.wmql_protocol import (OFFICIAL_DIR, official_query_names, canonical_wmql_selection,
                                       atlas_compatibility, bundle_assessment, CATALOG_PRESETS,
                                       CEREBELLAR_QUERIES, preset_query_names)


TRACT_QUERIER_REPOSITORY = "https://github.com/demianw/tract_querier"
VALID_TRACT_EXTENSIONS = {".vtk", ".vtp", ".trk"}
WMQL_QC_VERSION = 6
# Sampling sanity policy, not an anatomical accuracy or statistical-power test.
MIN_BUNDLE_STREAMLINES = 20
AUTHOR_QUERY_COMMIT = "5c69c594d489859e08485110678dca7f6f4cba40"
AUTHOR_ALIAS_SHA256 = "3fc2f3685af5061f5106f5f3c6c5644b926f0850211de0380082f274cb7be8a0"
AUTHOR_BUNDLE_QUERY_SHA256 = "554a4eed4b99cdbe497cc5296d49ea8a65961e2b1920c1868e41d2f0dfc080ba"


def _file_info(path: Path) -> dict[str, object]:
    expanded = path.expanduser()
    if not expanded.exists():
        return {"path": expanded.as_posix(), "exists": False}
    stat = expanded.stat()
    return {
        "path": expanded.resolve().as_posix(),
        "exists": True,
        "is_file": expanded.is_file(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


@dataclass(frozen=True)
class WMQLConfig:
    subject_id: str
    input_tractography: Path
    atlas_path: Path
    output_dir: Path
    query_path: Optional[Path] = None
    query_selection: tuple[str, ...] = ()
    tract_querier_root: Optional[Path] = None
    tract_querier_executable: Optional[Path] = None
    python_executable: Optional[Path] = None
    include_dirs: tuple[Path, ...] = ()
    threshold: float = 2.0
    length_threshold: float = 0.0
    execute: bool = False
    query_preset: str = "official"
    atlas_mode: str = 'native'


@dataclass(frozen=True)
class WMQLResult:
    subject_id: str
    output_dir: Path
    manifest_path: Path
    qc_path: Path
    command: tuple[str, ...]
    executed: bool
    backend_status: dict[str, Any]
    return_code: Optional[int] = None
    status: str = 'planned'


def bundled_wmql_query_path(preset: str = "official") -> Path:
    if preset == 'official':
        return OFFICIAL_DIR / 'freesurfer_queries.qry'
    if preset in ('cerebellar', 'official_cerebellar'):
        return OFFICIAL_DIR.parent / (preset + '.qry')
    # Read-only historical protocols remain addressable for reproducibility.
    files = {"ddparcel_act": "AF_CST_DDParcel_ACT.qry", "author": "AF_CST_ACT.qry",
             "published_endpoints": "AF_CST.qry"}
    if preset not in files:
        raise ValueError(f"Unknown WMQL query preset: {preset}")
    return Path(__file__).resolve().parent / "wmql_queries" / files[preset]


def wmql_query_dependencies(query_path: Path, include_dirs) -> dict[str, str]:
    """Follow the same ordered import lookup as the author parser."""
    records, active = {}, set()
    folders = [Path(directory).resolve() for directory in include_dirs]

    def visit(path):
        path = Path(path).resolve()
        if path in active:
            raise ValueError(f"Cyclic WMQL query import: {path}")
        if str(path) in records:
            return
        active.add(path)
        contents = path.read_bytes()
        records[str(path)] = hashlib.sha256(contents).hexdigest()
        for node in ast.walk(ast.parse(contents.decode('utf-8'), filename=str(path))):
            if not isinstance(node, ast.Import):
                continue
            for alias in node.names:
                imported = next((directory / alias.name for directory in folders if (directory / alias.name).is_file()), None)
                if imported is None:
                    raise FileNotFoundError(f"WMQL imported query not found: {alias.name}")
                visit(imported)
        active.remove(path)

    visit(query_path)
    return records


def wmql_query_cache_current(manifest) -> bool:
    try:
        sources = manifest.get('query_dependencies')
        snapshot = manifest.get('query_snapshot_dependencies')
        return (bool(sources) and bool(snapshot)
                and sources == wmql_query_dependencies(Path(manifest['query_path']), manifest['query_include_dirs'])
                and snapshot == wmql_query_dependencies(Path(manifest['staged_query']), [Path(manifest['staged_query']).parent]))
    except (OSError, ValueError, KeyError, TypeError, AttributeError, SyntaxError):
        return False


def wmql_query_provenance(query_path, dependencies=None):
    query_path = Path(query_path)
    digest = hashlib.sha256(Path(query_path).read_bytes()).hexdigest()
    for preset in ('cerebellar', 'official_cerebellar'):
        if query_path.resolve() == bundled_wmql_query_path(preset).resolve():
            return {'preset': preset, 'sha256': digest, 'extension_version': 1,
                    'extension_queries': list(CEREBELLAR_QUERIES),
                    'reference': 'https://tract-querier.readthedocs.io/en/latest/example_script.html',
                    'scope': 'SlicerDSpace regional cerebellar queries executed by the author WMQL engine; not published ICP/MCP/SCP definitions.',
                    'official_rules_included': preset == 'official_cerebellar'}
    published = bundled_wmql_query_path('published_endpoints').with_name('CST_WMQL2016.qry')
    endpoint_digests = {hashlib.sha256(path.read_bytes()).hexdigest() for path in (published, bundled_wmql_query_path('published_endpoints'))}
    author_selection = digest == hashlib.sha256(bundled_wmql_query_path('author').read_bytes()).hexdigest()
    adapted_selection = digest == hashlib.sha256(bundled_wmql_query_path('ddparcel_act').read_bytes()).hexdigest()
    if digest in endpoint_digests or digest == AUTHOR_BUNDLE_QUERY_SHA256 or author_selection or adapted_selection:
        data_dir = _default_query_data_dir()
        if dependencies is None:
            dependencies = wmql_query_dependencies(query_path, [query_path.parent] + ([data_dir] if data_dir else []))
        aliases = [value for path, value in dependencies.items() if Path(path).name == 'FreeSurfer.qry']
        # Installed extensions do not contain the optional author source tree.
        # Certify bytes against the pinned upstream file, not the local checkout.
        original_aliases = aliases == [AUTHOR_ALIAS_SHA256]
        author_query = digest == AUTHOR_BUNDLE_QUERY_SHA256 or author_selection
        if (author_selection or adapted_selection) and not any(Path(path).name == 'freesurfer_queries.qry' and value == AUTHOR_BUNDLE_QUERY_SHA256
                                        for path, value in dependencies.items()):
            return {'preset': 'ddparcel_act_custom_definitions' if adapted_selection and not author_selection else 'author_selection_custom_definitions', 'sha256': digest,
                    'scope': 'Author selection wrapper with modified or absent bundle definitions; not certified upstream.'}
        if not original_aliases:
            return {'preset': ('author_statement_custom_aliases' if author_query else
                               'ddparcel_act_custom_aliases' if adapted_selection else 'published_statement_custom_aliases'),
                    'sha256': digest,
                    'scope': 'Recognized query statement with modified or unverified regional aliases; not certified equivalent to the reference definition.'}
        source = f'{TRACT_QUERIER_REPOSITORY}/blob/{AUTHOR_QUERY_COMMIT}/tract_querier/data/'
        if adapted_selection and not author_query:
            return {'preset': 'ddparcel_af_candidates_author_cst', 'sha256': digest,
                    'reference': source + 'freesurfer_queries.qry',
                    'aliases_reference': source + 'FreeSurfer.qry',
                    'AF': 'Existing DDParcel-adapted cortical endpoint candidates, unchanged from AF_CST.qry; not the author crossing/exclusion AF definition',
                    'CST': {'definition': 'Unmodified author repository motor/brainstem crossing and exclusion predicates.'},
                    'scope': 'Explicit compatibility combination, not an unchanged official AF/CST protocol. No subject-specific fallback; sampling and anatomy remain separate.',
                    'act_compatibility': 'AF does not require cortical crossing percentage; CST does not require brainstem termination.'}
        if author_query:
            return {'preset': 'author_repository_crossing_exclusion', 'sha256': digest,
                    'reference': source + 'freesurfer_queries.qry',
                    'aliases_reference': source + 'FreeSurfer.qry',
                    'CST': {'definition': 'Brainstem and motor-region crossings with repository exclusions; not the published endpoint predicate.'},
                    'scope': 'Fixed author crossing/exclusion protocol. Distinct from the strict endpoint protocol; historical failures remain failures.',
                    'selection_wrapper': author_selection,
                    'act_compatibility': 'Requires brainstem traversal, not termination inside white matter.'}
        provenance = {'preset': 'published_cst_endpoints' if digest == hashlib.sha256(published.read_bytes()).hexdigest() else 'af_candidates_published_cst',
                'sha256': digest, 'aliases_reference': source + 'FreeSurfer.qry',
                'CST': {'definition': 'brainstem endpoint AND precentral/postcentral endpoint; original FreeSurfer aliases',
                        'reference': 'https://tract-querier.readthedocs.io/en/latest/example_script.html',
                        'paper': 'https://doi.org/10.1007/s00429-015-1179-4'},
                'scope': 'Definition provenance only; source membership and sampling do not certify anatomy.'}
        if provenance['preset'] == 'af_candidates_published_cst':
            provenance['AF'] = 'DDParcel-adapted endpoint candidates; not the legacy crossing/exclusion AF definition'
        return provenance
    return {'preset': 'external_query', 'sha256': digest,
            'scope': 'Explicit external rules, not assumed equivalent to the bundled or published endpoint definitions.'}


def _default_query_data_dir() -> Optional[Path]:
    candidate = Path(__file__).resolve().parent.parent / "third_party" / "tract_querier" / "tract_querier" / "data"
    return candidate if candidate.is_dir() else None


def _resolve_python(python_executable: Optional[Path]) -> Path:
    if python_executable is not None:
        candidate = python_executable.expanduser().absolute()
        if not candidate.is_file():
            raise FileNotFoundError(f"Python executable not found: {candidate}")
        return candidate
    for env_name in ("SLICERDSPACE_WMQL_PYTHON", "TRACT_QUERIER_PYTHON"):
        env_value = os.environ.get(env_name)
        if env_value:
            candidate = Path(env_value).expanduser().absolute()
            if not candidate.is_file():
                raise FileNotFoundError(f"Python executable not found from {env_name}: {candidate}")
            return candidate
    return Path(sys.executable).absolute()


def _probe_python_package(python_path: Path, tract_querier_root: Optional[Path]) -> dict[str, Any]:
    env = python_tool_environment(python_path)
    if tract_querier_root is not None:
        env["PYTHONPATH"] = tract_querier_root.expanduser().resolve().as_posix() + os.pathsep + env.get("PYTHONPATH", "")
    probe = (
        "import importlib.util, json; "
        "print(json.dumps({'tract_querier': importlib.util.find_spec('tract_querier') is not None, "
        "'nibabel': importlib.util.find_spec('nibabel') is not None, "
        "'numpy': importlib.util.find_spec('numpy') is not None}))"
    )
    try:
        completed = subprocess.run(
            [python_path.as_posix(), "-c", probe],
            check=True,
            capture_output=True,
            env=env,
            **text_subprocess_kwargs(),
        )
        return json.loads(completed.stdout.strip() or "{}")
    except Exception as exc:
        return {"error": str(exc), "tract_querier": False, "nibabel": False, "numpy": False}


def inspect_wmql_backend(
    *,
    tract_querier_root: Optional[Path] = None,
    tract_querier_executable: Optional[Path] = None,
    python_executable: Optional[Path] = None,
) -> dict[str, Any]:
    """Inspect Tract Querier availability without running WMQL."""

    python_path = _resolve_python(python_executable)
    root = tract_querier_root.expanduser().resolve() if tract_querier_root is not None else None
    if root is None:
        env_root = os.environ.get("SLICERDSPACE_TRACT_QUERIER_ROOT") or os.environ.get("TRACT_QUERIER_ROOT")
        root = Path(env_root).expanduser().resolve() if env_root else None
    explicit_cli = tract_querier_executable.expanduser().resolve() if tract_querier_executable is not None else None
    env_cli = os.environ.get("SLICERDSPACE_WMQL_EXECUTABLE") or os.environ.get("TRACT_QUERIER_EXECUTABLE")
    cli_path: Optional[Path] = None
    if explicit_cli is not None:
        cli_path = explicit_cli
    elif env_cli:
        cli_path = Path(env_cli).expanduser().resolve()
    else:
        found = shutil.which("tract_querier")
        if found:
            cli_path = Path(found).resolve()

    if root is None and cli_path is None:
        from shared.backend_sources import available_source
        root = available_source("wmql")

    root_script = root / "scripts" / "cli_tract_querier.py" if root is not None else None
    package_status = _probe_python_package(python_path, root)
    commands = {
        "tract_querier_executable": cli_path.as_posix() if cli_path and cli_path.is_file() else None,
        "root_script": root_script.as_posix() if root_script and root_script.is_file() else None,
        "python": python_path.as_posix(),
    }
    missing: list[str] = []
    has_runner = bool(commands["tract_querier_executable"] or commands["root_script"])
    if not has_runner:
        missing.append("tract_querier executable or tract_querier_root/scripts/cli_tract_querier.py")
    for package in ("tract_querier", "nibabel", "numpy"):
        if not package_status.get(package):
            missing.append(f"python package: {package}")

    return {
        "kind": "wmql_backend",
        "repository": TRACT_QUERIER_REPOSITORY,
        "available": not missing,
        "root": root.as_posix() if root else None,
        "commands": commands,
        "python_packages": package_status,
        "missing_required": missing,
    }


def _query_include_dirs(config: WMQLConfig, query_path: Path) -> list[Path]:
    dirs: list[Path] = [query_path.parent]
    dirs.extend(path.expanduser().resolve() for path in config.include_dirs)
    dirs.append(OFFICIAL_DIR)
    if config.tract_querier_root is not None:
        data_dir = config.tract_querier_root.expanduser().resolve() / "tract_querier" / "data"
        if data_dir.is_dir():
            dirs.append(data_dir)
    default_data = _default_query_data_dir()
    if default_data is not None:
        dirs.append(default_data.resolve())
    unique: list[Path] = []
    for path in dirs:
        if path not in unique:
            unique.append(path)
    return unique


def _requires_windows_query_staging() -> bool:
    """TractQuerier splits ``-I`` on ``:`` and therefore breaks drive paths."""

    return os.name == "nt"


def _stage_query_includes(include_dirs: list[Path], queries_dir: Path) -> None:
    """Stage query includes in one cwd that TractQuerier can search on Windows."""

    staged_names = set()
    for include_dir in include_dirs:
        source_dir = include_dir.expanduser().resolve()
        if not source_dir.is_dir():
            raise FileNotFoundError(f"WMQL include directory not found: {source_dir}")
        for source in sorted(source_dir.glob("*.qry")):
            target = queries_dir / source.name
            if source.name in staged_names:
                continue
            staged_names.add(source.name)
            if source.resolve() == target.resolve():
                continue
            # Keep first-directory precedence within this invocation's snapshot.
            shutil.copy2(source, target)


def build_wmql_command(config: WMQLConfig, backend_status: Optional[dict[str, Any]] = None) -> tuple[str, ...]:
    backend = backend_status or inspect_wmql_backend(
        tract_querier_root=config.tract_querier_root,
        tract_querier_executable=config.tract_querier_executable,
        python_executable=config.python_executable,
    )
    if backend.get("root"):
        config = replace(config, tract_querier_root=Path(backend["root"]))
    query_path = (config.query_path or bundled_wmql_query_path(config.query_preset)).expanduser().resolve()
    output_prefix = config.output_dir.expanduser().resolve() / f"{config.subject_id}_desc-wmql"
    include_dirs = _query_include_dirs(config, query_path)
    query_selection = ",".join(name.lower() for name in config.query_selection if name.strip())

    cli = backend["commands"].get("tract_querier_executable")
    root_script = backend["commands"].get("root_script")
    if cli:
        cmd = [cli]
    elif root_script:
        cmd = [backend["commands"]["python"], root_script]
    else:
        cmd = [backend["commands"]["python"], "-m", "tract_querier.scripts.cli_tract_querier"]

    cmd.extend(
        [
            "-t",
            config.input_tractography.expanduser().resolve().as_posix(),
            "-a",
            config.atlas_path.expanduser().resolve().as_posix(),
            "-q",
            query_path.as_posix(),
            "-o",
            output_prefix.as_posix(),
        ]
    )
    if include_dirs and not _requires_windows_query_staging():
        # TractQuerier currently hard-codes ':' as its include separator.
        cmd.extend(["-I", ":".join(path.as_posix() for path in include_dirs)])
    cmd.extend(
        [
            "--threshold",
            str(config.threshold),
            "--length_threshold",
            str(config.length_threshold),
        ]
    )
    if query_selection:
        cmd.extend(["--query_selection", query_selection])
    return tuple(cmd)


def summarize_wmql_outputs(output_dir: Path, subject_id: Optional[str] = None) -> dict[str, Any]:
    root = output_dir.expanduser().resolve()
    bundle_files = sorted(
        path for path in root.glob("*") if path.suffix.lower() in VALID_TRACT_EXTENSIONS
        and "_desc-wmql_" in path.name
        and (subject_id is None or path.name.startswith(f'{subject_id}_desc-wmql_'))
    ) if root.is_dir() else []
    bundle_details: list[dict[str, Any]] = []
    for path in bundle_files:
        detail: dict[str, Any] = {"path": path.as_posix(), "size": path.stat().st_size,
                                  "query_name": path.stem.split('_desc-wmql_', 1)[1].lower()}
        if path.suffix.lower() in {".trk", ".tck"}:
            try:
                import nibabel as nib
                import numpy as np
                count = 0
                for track in nib.streamlines.load(path.as_posix(), lazy_load=True).streamlines:
                    if len(track) < 2 or not np.isfinite(track).all() or not np.any(track != track[0]):
                        raise ValueError('Non-finite, degenerate or incomplete TRK streamline geometry')
                    count += 1
                detail["streamline_count"] = count
            except Exception as exc:  # pragma: no cover - best-effort QC
                detail["streamline_count_error"] = str(exc)
        elif path.suffix.lower() in {".vtk", ".vtp"}:
            count = polydata_streamline_count(path)
            if count is None:
                detail['streamline_count_error'] = 'Unable to decode complete finite VTK/VTP streamline geometry'
            else:
                detail['streamline_count'] = count
        bundle_details.append(detail)
    readable = [item for item in bundle_details if "streamline_count" in item]
    nonempty = [item for item in readable if int(item["streamline_count"]) > 0]
    return {
        "root": root.as_posix(),
        "root_exists": root.is_dir(),
        "bundle_count": len(bundle_files),
        "bundle_files": [path.as_posix() for path in bundle_files],
        "bundles": bundle_details,
        "bundle_readable_count": len(readable),
        "bundle_nonempty_count": len(nonempty),
        "streamline_count": sum(int(item["streamline_count"]) for item in readable),
        "unreadable_bundle_files": [
            str(item["path"]) for item in bundle_details if "streamline_count_error" in item
        ],
        "empty_bundle_files": [
            str(item["path"])
            for item in readable
            if int(item["streamline_count"]) == 0
        ],
    }


def _wmql_output_contract_error(
    output_summary: dict[str, Any],
    expected_bundle_count: int,
    requested_queries: tuple[str, ...] = (),
    *,
    require_sampling: bool = True,
) -> Optional[str]:
    expected_names = {name.strip().lower() for name in requested_queries if name.strip()}
    present = {item.get('query_name') for item in output_summary.get('bundles', [])
               if int(item.get('streamline_count', 0)) > 0}
    missing = expected_names - present
    if missing and require_sampling:
        return 'WMQL is missing requested non-empty query outputs: ' + ', '.join(sorted(missing))
    bundle_count = int(output_summary.get("bundle_count", 0) or 0)
    readable_count = int(output_summary.get("bundle_readable_count", 0) or 0)
    nonempty_count = int(output_summary.get("bundle_nonempty_count", 0) or 0)
    expected = max(1, expected_bundle_count) if require_sampling else 1
    unexpected = present - expected_names if expected_names else set()
    if unexpected:
        return 'WMQL returned unrequested queries: ' + ', '.join(sorted(unexpected))
    if bundle_count < expected:
        return f"WMQL produced {bundle_count} bundle files; at least {expected} were requested."
    if readable_count != bundle_count:
        return "WMQL produced one or more bundle files that could not be decoded."
    if nonempty_count < expected:
        return f"WMQL produced only {nonempty_count} non-empty bundles; at least {expected} were requested."
    sparse = [f"{item['query_name']}={item['streamline_count']}"
              for item in output_summary.get('bundles', [])
              if item.get('query_name') in (expected_names or present)
              and int(item.get('streamline_count', 0)) < MIN_BUNDLE_STREAMLINES]
    if sparse and require_sampling:
        return (f"WMQL insufficient bundle sampling (local minimum {MIN_BUNDLE_STREAMLINES} streamlines): "
                + ', '.join(sparse) + '. Review atlas label compatibility and tractography coverage; '
                'do not interpret these candidates as validated bundle quantification.')
    return None


def _validate_wmql_config(config: WMQLConfig, backend_status: dict[str, Any]) -> None:
    if config.atlas_mode not in ('native', 'ddparcel_wm'):
        raise ValueError('Unknown WMQL atlas mode: ' + config.atlas_mode)
    if config.atlas_mode != 'native' and (config.query_path is not None or config.query_preset not in CATALOG_PRESETS):
        raise ValueError('Experimental DDParcel WM preparation requires a bundled catalogue, not custom/historical rules')
    if not config.subject_id.strip():
        raise ValueError("subject_id is required.")
    if config.input_tractography.suffix.lower() not in VALID_TRACT_EXTENSIONS:
        raise ValueError("WMQL input tractography must be .vtk, .vtp, or .trk.")
    if not config.input_tractography.expanduser().is_file():
        raise FileNotFoundError(f"Input tractography not found: {config.input_tractography}")
    if not config.atlas_path.expanduser().is_file():
        raise FileNotFoundError(f"WMQL atlas/parcellation not found: {config.atlas_path}")
    query_path = config.query_path or bundled_wmql_query_path(config.query_preset)
    if not query_path.expanduser().is_file():
        raise FileNotFoundError(f"WMQL query file not found: {query_path}")
    import math
    if not math.isfinite(config.threshold) or not 0 <= config.threshold <= 100:
        raise ValueError("threshold must be finite and between 0 and 100.")
    if not math.isfinite(config.length_threshold) or config.length_threshold < 0:
        raise ValueError("length_threshold must be non-negative.")
    if config.execute and not backend_status["available"]:
        raise RuntimeError("WMQL backend is incomplete. Missing: " + ", ".join(backend_status["missing_required"]))


@scheduled_stage('wmql')
def run_wmql_bundle_segmentation(
    config: WMQLConfig,
    *,
    log: Optional[Callable[[str], None]] = None,
) -> WMQLResult:
    """Prepare or execute WMQL bundle extraction for one whole-brain tractogram."""

    official_mode = config.query_preset == 'official' and config.query_path is None
    catalogue_mode = config.query_preset in CATALOG_PRESETS and config.query_path is None
    cerebellar_mode = catalogue_mode and config.query_preset != 'official'
    if catalogue_mode:
        config = replace(config, query_selection=canonical_wmql_selection(config.query_selection, config.query_preset))
    requested_queries = (config.query_selection or preset_query_names(config.query_preset)) if catalogue_mode else config.query_selection
    backend_status = inspect_wmql_backend(
        tract_querier_root=config.tract_querier_root,
        tract_querier_executable=config.tract_querier_executable,
        python_executable=config.python_executable,
    )
    if backend_status.get("root"):
        config = replace(config, tract_querier_root=Path(backend_status["root"]))
    _validate_wmql_config(config, backend_status)
    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    queries_root = output_dir / "queries"
    queries_root.mkdir(parents=True, exist_ok=True)
    queries_dir = Path(tempfile.mkdtemp(prefix="run-", dir=queries_root))
    query_path = (config.query_path or bundled_wmql_query_path(config.query_preset)).expanduser().resolve()
    include_dirs = _query_include_dirs(config, query_path)
    query_dependencies = wmql_query_dependencies(query_path, include_dirs)
    provenance = wmql_query_provenance(query_path, query_dependencies)
    if catalogue_mode:
        if ((config.query_preset != 'cerebellar' and not any(
                Path(path).name == 'freesurfer_queries.qry' and digest == AUTHOR_BUNDLE_QUERY_SHA256
                for path, digest in query_dependencies.items()))
                or not any(Path(path).name == 'FreeSurfer.qry' and digest == AUTHOR_ALIAS_SHA256
                           for path, digest in query_dependencies.items())):
            raise ValueError('Official WMQL rule bytes do not match the pinned author source')
    if official_mode:
        provenance.update(preset='official_multibundle', upstream_commit=AUTHOR_QUERY_COMMIT,
                          scope='Unmodified author repository rules and label aliases; no DDParcel-specific predicate overrides')
    source_atlas = config.atlas_path
    effective_atlas, atlas_preparation = source_atlas, None
    compatibility, compatibility_error = None, None
    if catalogue_mode and config.execute:
        import nibabel as nib
        try:
            if config.atlas_mode == 'ddparcel_wm':
                from tractography.wmql_atlas import prepare_ddparcel_atlas
                effective_atlas, atlas_preparation = prepare_ddparcel_atlas(source_atlas, output_dir / 'atlas_cache')
                if log:
                    log('[WMQL] Experimental 5-mm adjacent-WM atlas prepared; native DDParcel labels retained separately')
            compatibility = atlas_compatibility(effective_atlas)
            if cerebellar_mode:
                missing = sorted({7, 8, 16, 46, 47} - set(compatibility['label_ids']))
                if missing:
                    raise ValueError('Cerebellar queries require native cerebellar WM/cortex and brainstem labels: '
                                     + ', '.join(map(str, missing)))
                if config.query_preset == 'cerebellar':
                    # Regional selections do not require regional cerebral WM labels.
                    compatibility['warnings'] = []
        except (OSError, ValueError, nib.filebasedimages.ImageFileError) as exc:
            compatibility_error = f'Atlas compatibility failed: {exc}'
            compatibility = dict(status='failed', error=compatibility_error)
    staged_query = queries_dir / query_path.name
    if query_path.resolve() != staged_query.resolve():
        shutil.copy2(query_path, staged_query)
    _stage_query_includes(include_dirs, queries_dir)
    snapshot_dependencies = wmql_query_dependencies(staged_query, [queries_dir])
    if {Path(path).name: digest for path, digest in query_dependencies.items()} != {
        Path(path).name: digest for path, digest in snapshot_dependencies.items()
    }:
        raise ValueError('WMQL staged imports differ from the selected query source')
    command_config = replace(config, query_path=staged_query, atlas_path=effective_atlas)
    command = build_wmql_command(command_config, backend_status)

    manifest_path = output_dir / "wmql_manifest.json"
    qc_path = output_dir / "wmql_qc.json"
    previous_records = archive_run_records(output_dir, ('wmql_manifest.json', 'wmql_qc.json')) if config.execute else None
    output_summary = summarize_wmql_outputs(output_dir, config.subject_id)
    manifest: dict[str, Any] = {
        "kind": "wmql_bundle_segmentation",
        "version": WMQL_QC_VERSION,
        "subject_id": config.subject_id,
        "query_preset": config.query_preset,
        "input_tractography": config.input_tractography.expanduser().resolve().as_posix(),
        "atlas_path": effective_atlas.expanduser().resolve().as_posix(),
        "source_atlas_path": source_atlas.expanduser().resolve().as_posix(),
        "atlas_mode": config.atlas_mode,
        "atlas_preparation": atlas_preparation,
        "query_path": query_path.as_posix(),
        "query_sha256": hashlib.sha256(query_path.read_bytes()).hexdigest(),
        "query_dependencies": query_dependencies,
        "query_snapshot_dependencies": snapshot_dependencies,
        "query_include_dirs": list(map(str, include_dirs)),
        "query_provenance": provenance,
        "staged_query": staged_query.as_posix(),
        "query_selection": list(config.query_selection),
        "requested_queries": list(requested_queries),
        "official_multibundle": official_mode,
        "coverage_report_mode": catalogue_mode,
        "atlas_compatibility": compatibility,
        "threshold": config.threshold,
        "length_threshold": config.length_threshold,
        "executed": bool(config.execute),
        "status": "planned" if not config.execute else "running",
        "command": list(command),
        "execution_cwd": queries_dir.as_posix() if _requires_windows_query_staging() else None,
        "backend": backend_status,
        "previous_records": previous_records,
        "dependencies": {
            "input_tractography": _file_info(config.input_tractography),
            "atlas": _file_info(source_atlas),
            "effective_atlas": _file_info(effective_atlas),
            "query": _file_info(query_path),
        },
        "outputs": {"qc": qc_path.as_posix(), "summary": output_summary},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if log:
        mode = "Executing" if config.execute else "Prepared dry-run"
        log(f"[WMQL] {mode} command: {' '.join(command)}")
        log(f"[WMQL] Manifest: {manifest_path}")

    start_time = time.monotonic()
    return_code: Optional[int] = None
    execution_error: Optional[str] = compatibility_error
    execution_exception: Optional[Exception] = None
    staging = None
    published_archive = None
    if config.execute:
        env = python_tool_environment(backend_status['commands']['python'])
        if config.tract_querier_root is not None:
            root = config.tract_querier_root.expanduser().resolve()
            env["PYTHONPATH"] = root.as_posix() + os.pathsep + env.get("PYTHONPATH", "")
        try:
            if compatibility_error:
                raise ValueError(compatibility_error)
            staging = new_product_staging(output_dir, 'wmql')
            runtime_command = list(command)
            runtime_command[runtime_command.index('-o') + 1] = str(staging / f'{config.subject_id}_desc-wmql')
            manifest['executed_command'] = runtime_command
            completed = subprocess.run(
                runtime_command,
                check=True,
                env=env,
                cwd=queries_dir if _requires_windows_query_staging() else None,
            )
            return_code = completed.returncode
        except subprocess.CalledProcessError as exc:
            return_code = exc.returncode
            execution_error = str(exc)
            execution_exception = exc
        except OSError as exc:
            # Preserve launch failures (missing/broken executables, permission
            # errors) in QC/manifest before surfacing the original exception.
            execution_error = f"{type(exc).__name__}: {exc}"
            execution_exception = exc
        except ValueError as exc:
            execution_error = str(exc)
            execution_exception = exc
        finally:
            output_summary = summarize_wmql_outputs(staging, config.subject_id) if staging else {}
            if not wmql_query_cache_current(manifest):
                execution_error = 'WMQL query/imports changed during execution; outputs are not reusable'
            if any(record != _file_info(Path(record['path'])) for record in manifest['dependencies'].values()):
                execution_error = 'WMQL source inputs changed during execution; outputs are not reusable'
            if execution_error is None:
                execution_error = _wmql_output_contract_error(output_summary, len(requested_queries), requested_queries,
                                                               require_sampling=not catalogue_mode)
            if staging:
                try:
                    if execution_error is None:
                        previous_names = [p.name for p in output_dir.glob(f'{config.subject_id}_desc-wmql_*')
                                          if p.suffix.lower() in VALID_TRACT_EXTENSIONS]
                        published_archive = publish_products(staging, output_dir, previous_names)
                        output_summary = summarize_wmql_outputs(output_dir, config.subject_id)
                    else:
                        staging = retain_failed_products(staging, output_dir)
                        output_summary = summarize_wmql_outputs(staging, config.subject_id)
                except OSError as exc:
                    execution_error = f'Failed to publish WMQL products: {exc}'

    if config.execute and execution_error is None:
        execution_error = _wmql_output_contract_error(
            output_summary,
            len(requested_queries), requested_queries, require_sampling=not catalogue_mode,
        )

    elapsed_seconds = time.monotonic() - start_time
    assessment = bundle_assessment(output_summary, requested_queries, MIN_BUNDLE_STREAMLINES)
    coverage_warning = bool(catalogue_mode and not assessment['all_requested_sampling_passed'])
    scientific_warnings = [f"{b['query_name']}: {b['state']} ({b['streamline_count']} streamlines)"
                           for b in assessment['bundles'] if not b['sampling_passed']]
    scientific_warnings.extend((compatibility or {}).get('warnings', []))
    if atlas_preparation:
        scientific_warnings.append('Experimental adjacent-WM territories: review anatomical boundaries and bundle identity before quantitative use.')
    qc = {
        "kind": "wmql_bundle_segmentation_qc",
        "version": WMQL_QC_VERSION,
        "subject_id": config.subject_id,
        "status": "failed" if execution_error else ("completed" if config.execute else "planned"),
        "executed": bool(config.execute),
        "return_code": return_code,
        "elapsed_seconds": elapsed_seconds,
        "human_review": "pending",
        "execution_passed": bool(config.execute and execution_error is None),
        "output_integrity_passed": bool(config.execute and execution_error is None),
        "automatic_qc_passed": bool(config.execute and execution_error is None and not coverage_warning
                                    and not (compatibility or {}).get('warnings') and not atlas_preparation),
        "atlas_mode": config.atlas_mode,
        "atlas_preparation": atlas_preparation,
        "official_multibundle": official_mode,
        "coverage_report_mode": catalogue_mode,
        "bundle_assessment": assessment,
        "atlas_compatibility": compatibility,
        "quantification_eligible": False,
        "acceptance_criteria": {"minimum_streamlines_per_requested_bundle": MIN_BUNDLE_STREAMLINES,
                                "scope": "Sampling sanity only; passing does not validate anatomical identity."},
        "scientific_status": ("insufficient_or_invalid_products" if execution_error else
                              "partial_bundle_coverage" if coverage_warning else "anatomical_review_pending"),
        "scientific_warnings": scientific_warnings,
        "previous_products": published_archive,
        "query_provenance": provenance,
        "scientific_scope": "Requested-query identity, frozen imports and decoded geometry; not expert anatomical validation. Query definitions and references are recorded separately.",
        "backend_available": bool(backend_status["available"]),
        "output_summary": output_summary,
        "error": execution_error,
    }
    qc_path.write_text(json.dumps(qc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest["status"] = qc["status"]
    manifest["executed"] = bool(config.execute)
    manifest["return_code"] = return_code
    manifest["runtime_seconds"] = elapsed_seconds
    manifest["error"] = execution_error
    manifest["previous_products"] = published_archive
    manifest["outputs"]["summary"] = output_summary
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if execution_exception is not None:
        raise execution_exception

    return WMQLResult(
        subject_id=config.subject_id,
        output_dir=output_dir,
        manifest_path=manifest_path,
        qc_path=qc_path,
        command=command,
        executed=bool(config.execute),
        backend_status=backend_status,
        return_code=return_code,
        status=qc['status'],
    )
