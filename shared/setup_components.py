"""Components, sizes and licences listed by the one-confirmation setup dialog and CLI (B7).

A licence version is the SHA-256 of its local licence file. SPEC-GAP: a licence without a local
file (FSL, MRtrix3, PNL until its file ships) uses the SHA-256 of its URL string instead.
"""

import hashlib
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FREESURFER_TERMS_URL = "https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSoftwareLicense"
RESOURCE_TERMS_URL = "https://zenodo.org/record/2648292"


def _licences():
    from shared.backend_sources import SOURCES
    from shared.managed_pnl import PNL_TERMS_URL
    from shared.managed_runtimes import FSL_TERMS_URL

    def source(name, title, suffix, files=()):
        spec = SOURCES[name]
        return dict(name=title, files=files,
                    url="https://github.com/%s/blob/%s/%s" % (spec["repository"], spec["revision"], suffix))

    return {
        "fsl": dict(name="FSL licence", url=FSL_TERMS_URL, files=()),
        "pnl": dict(name="PNL CNN brain mask: Brigham Contribution and Software License Agreement",
                    url=PNL_TERMS_URL, files=("shared/pnl_cnn/LICENSE.txt",)),
        "freesurfer": dict(name="FreeSurfer software licence", url=FREESURFER_TERMS_URL, files=("licenses/FreeSurfer.txt",)),
        "wmql": source("wmql", "tract_querier licence", "license.rst"),
        "wma": source("wma", "WhiteMatterAnalysis licence", "License.txt", ("licenses/WMA.txt",)),
        "tractcloud": source("tractcloud", "TractCloud licence", "LICENSE"),
        "optional_resources": dict(name="ORG atlas, TractCloud data and brain-mask model terms", url=RESOURCE_TERMS_URL,
                                   files=("docs/assets/EXTERNAL_RESOURCES.md",)),
    }


def licence_file(licence, root=REPO_ROOT):
    """The first local licence file that exists, or None."""
    return next((Path(root) / name for name in licence.get("files", ()) if (Path(root) / name).is_file()), None)


def licence_version(licence, root=REPO_ROOT):
    path = licence_file(licence, root)
    data = path.read_bytes() if path else str(licence["url"]).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def licence_record(licence_id, root=REPO_ROOT):
    licence = _licences()[licence_id]
    path = licence_file(licence, root)
    return dict(id=licence_id, name=licence["name"], url=licence["url"], file=str(path) if path else None,
                version=licence_version(licence, root))


def _component(component_id, title, size, licence_id=None, root=REPO_ROOT):
    return dict(id=component_id, title=title, size_bytes=size,
                licence=licence_record(licence_id, root) if licence_id else None)


def plan_components(plan, root=REPO_ROOT):
    """Each component of the selected closure with size, licence name, link and local licence file."""
    items = []
    for name in plan.get("backend_profiles", ()):
        from shared.backend_sources import SOURCES
        items.append(_component("backend:" + name, "Algorithm source: " + name, SOURCES[name]["bytes"], name, root))
    if plan.get("python_packages"):
        items.append(_component("python_packages", "Python packages: " + ", ".join(plan["python_packages"]), None))
    if plan.get("asset_profiles"):
        items.append(_component("model_assets", "Model assets: " + ", ".join(plan["asset_profiles"]),
                                plan.get("core_model_profile_bytes") or None))
    if plan.get("optional_asset_profiles"):
        items.append(_component("optional_resources", "Resources: " + ", ".join(plan["optional_asset_profiles"]),
                                plan.get("optional_asset_size_bytes") or None, "optional_resources", root))
    windows = plan.get("windows_runtime") or {}
    if windows.get("command"):
        items.append(_component("windows_runtime", "WSL runtime (%s): MRtrix3 and FSL" % windows["distribution"], None, "fsl", root))
        if windows.get("installs_freesurfer"):
            items.append(_component("windows_freesurfer", "FreeSurfer (WSL)", None, "freesurfer", root))
    else:
        for name in plan.get("managed_profiles", ()):
            licence = "pnl" if name == "pnl_cnn" else "fsl" if name.startswith("fsl_") else None
            items.append(_component("managed:" + name, "Managed tool: " + name, None, licence, root))
    if plan.get("surface_prebuilt_required") or plan.get("surface_build_required"):
        items.append(_component("surface_runtime", "Surface components", plan.get("surface_download_bytes") or None,
                                "freesurfer", root))
    return items


def required_licences(components):
    """{licence id: record} for every licence the components need, in first-use order."""
    found = {}
    for component in components:
        licence = component.get("licence")
        if licence:
            found.setdefault(licence["id"], licence)
    return found


def pending_licences(required, accepted):
    """The required licences whose current version is not yet accepted."""
    def accepted_version(key):
        value = (accepted or {}).get(key)
        return value.get("version") if isinstance(value, dict) else value

    return {key: record for key, record in required.items() if accepted_version(key) != record["version"]}


def accept_licences(records, *, now=None):
    """Store accepted licence ids, versions and the time in the user config (B7)."""
    from shared.user_config import update_user_config

    stamp = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds").replace("+00:00", "Z")
    update_user_config({"accepted_licences": {key: dict(id=key, version=record["version"], time=stamp)
                                              for key, record in records.items()}})
    return stamp
