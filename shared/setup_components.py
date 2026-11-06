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
MRTRIX_TERMS_URL = "https://github.com/MRtrix3/mrtrix3/blob/3.0.8/LICENCE.txt"
MODEL_ASSETS_URL = "https://github.com/ChengjinLii/SlicerDSpace-models"
TRACTCLOUD_DATA_URL = "https://github.com/SlicerDMRI/TractCloud/releases/tag/v1.0.0"
DCM2NIIX_TERMS_URL = "https://github.com/rordenlab/dcm2niix/blob/master/license.txt"
PYPI_URL = "https://pypi.org/"


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
        # S-B8: the rest of the inventory, from the records the repo already keeps (MODEL_ASSET_TERMS.md, the ps1 pins).
        "mrtrix": dict(name="MRtrix3 licence (MPL-2.0)", url=MRTRIX_TERMS_URL, files=()),
        "model_assets": dict(name="Model asset terms: DDSurfer, DDSeg and DDParcel weights and templates", url=MODEL_ASSETS_URL,
                             files=("docs/assets/MODEL_ASSET_TERMS.md",)),
        "tractcloud_data": dict(name="TractCloud trained model and training data: Slicer/BWH agreement and upstream terms",
                                url=TRACTCLOUD_DATA_URL, files=("docs/assets/MODEL_ASSET_TERMS.md",)),
        "dcm2niix": dict(name="dcm2niix licence (BSD-style; bundled files carry their own)", url=DCM2NIIX_TERMS_URL, files=()),
        "pypi_packages": dict(name="PyPI package licences (each package states its own terms)", url=PYPI_URL, files=()),
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
    """Each component of the selected closure with size, licence name, link and local licence file (S-B8).

    Every component the steps of setup_steps install has an entry, and every entry names a licence.
    """
    from shared.backend_sources import SOURCES

    items = []
    for name in plan.get("backend_profiles", ()):
        items.append(_component("backend:" + name, "Algorithm source: " + name, SOURCES[name]["bytes"], name, root))
    packages = list(plan.get("python_packages", ()))
    pinned = [value for value in packages if str(value).lower().replace("_", "-").startswith("dcm2niix")]
    others = [value for value in packages if value not in pinned]
    if others:
        items.append(_component("python_packages", "Python packages: " + ", ".join(others), None, "pypi_packages", root))
    if pinned:
        items.append(_component("python:dcm2niix", "Python package: " + ", ".join(pinned), None, "dcm2niix", root))
    if plan.get("asset_profiles"):
        items.append(_component("model_assets", "Model assets: " + ", ".join(plan["asset_profiles"]),
                                plan.get("core_model_profile_bytes") or None, "model_assets", root))
    if plan.get("optional_asset_profiles"):
        items.append(_component("optional_resources", "Resources: " + ", ".join(plan["optional_asset_profiles"]),
                                plan.get("optional_asset_size_bytes") or None, "optional_resources", root))
    windows = plan.get("windows_runtime") or {}
    if windows.get("command"):
        # The ps1 installs MRtrix3, FSL, the three algorithm sources and the TractCloud data in WSL every time;
        # the ORG atlases unless -SkipLargeDownloads; FreeSurfer only with a licence file.
        where = "WSL runtime (%s)" % windows.get("distribution")
        items.append(_component("windows_runtime", where + ": MRtrix3", None, "mrtrix", root))
        items.append(_component("windows_fsl", where + ": FSL eddy and topup", None, "fsl", root))
        for name in ("wmql", "wma", "tractcloud"):
            items.append(_component("windows_backend:" + name, where + ": algorithm source " + name,
                                    SOURCES[name]["bytes"], name, root))
        items.append(_component("windows_tractcloud_data", where + ": TractCloud model and data", None, "tractcloud_data", root))
        if "-SkipLargeDownloads" not in windows["command"]:
            items.append(_component("windows_org_atlas", where + ": ORG atlases", None, "optional_resources", root))
        if windows.get("installs_freesurfer"):
            items.append(_component("windows_freesurfer", where + ": FreeSurfer", None, "freesurfer", root))
    else:
        for name in plan.get("managed_profiles", ()):
            licence = ("pnl" if name == "pnl_cnn" else "fsl" if name.startswith("fsl_") else
                       "mrtrix" if name == "mrtrix" else None)
            items.append(_component("managed:" + name, "Managed tool: " + name, None, licence, root))
    if plan.get("surface_prebuilt_required") or plan.get("surface_build_required"):
        items.append(_component("surface_runtime", "Surface components", plan.get("surface_download_bytes") or None,
                                "freesurfer", root))
    return items


# Step id -> prefixes of the component ids that cover what the step installs.
STEP_COMPONENTS = {"backend_sources": ("backend:",), "python_packages": ("python_packages", "python:"),
                   "model_assets": ("model_assets",), "optional_resources": ("optional_resources",),
                   "windows_runtime": ("windows_",), "managed_runtime": ("managed:",), "surface_runtime": ("surface_runtime",)}


def unlisted_licences(steps, components):
    """Names of the steps or components that install something without a listed licence (S-B8)."""
    missing = [item["id"] for item in components if not item.get("licence")]
    for step in steps:
        prefixes = STEP_COMPONENTS.get(step["id"])
        if prefixes and not any(item["id"].startswith(prefixes) for item in components):
            missing.append("step:" + step["id"])
    return missing


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
