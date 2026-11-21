"""Atlas-specific layouts and tract inventories for the WMA author pipeline."""

from pathlib import Path
import re

ATLAS_CHOICES = ("org", "east-west", "naba")
NABA_HEMISPHERIC = (
    "T_AF", "T_CB_D", "T_CB_V", "T_EmC", "T_ILF", "T_IOFF", "T_MdLF",
    "T_SLF-I", "T_SLF-II", "T_SLF-III", "T_UF", "T_ICP", "T_Intra-CBLM-I&P",
    "T_Intra-CBLM-PaT", "T_SCP", "T_CPC", "T_CST", "T_CR-F", "T_CR-P", "T_SF", "T_SO",
    "T_SP", "T_TF", "T_TO", "T_TT", "T_TP", "T_EC", "T_Sup-F", "T_Sup-FP",
    "T_Sup-O", "T_Sup-OT", "T_Sup-P", "T_Sup-PO", "T_Sup-PT", "T_Sup-T",
)
CALLOSAL = tuple("T_CC" + str(index) for index in range(1, 8))
ATLAS_DEFINITIONS = {
    "org": {
        "label": "ORG (adult / lifespan reference)",
        "directories": ("ORG-RegAtlas-100HCP", "ORG-800FC-100HCP"),
        "profile": "wma", "directory": "bundles/wma",
        "paper": "https://doi.org/10.1016/j.neuroimage.2018.06.027",
        "source": "https://doi.org/10.5281/zenodo.2648292",
    },
    "east-west": {
        "label": "East-West (Eastern / Western adults)",
        "directories": ("East-West_WM_Atlas_Reg", "East-West_WM_Atlas_800FC"),
        "profile": "wma-east-west", "directory": "bundles/wma/east-west",
        "paper": "https://doi.org/10.1038/s41597-024-03624-2",
        "source": "https://doi.org/10.57760/sciencedb.17354",
        "hemispheric": tuple(name.replace("T_Intra-CBLM", "T_CBLM") for name in NABA_HEMISPHERIC
                             if name not in ("T_CB_D", "T_CB_V")) + ("T_CB",),
        "commissural": CALLOSAL + ("T_MCP",),
    },
    "naba": {
        "label": "NABA (neonates / adults)",
        "directories": ("NABA-RegAtlas", "NABA-800FC"),
        "profile": "wma-naba", "directory": "bundles/wma/naba",
        "paper": "https://doi.org/10.1016/j.fmre.2025.12.030",
        "source": "https://doi.org/10.5281/zenodo.18227691",
        "grouping_source": "https://github.com/azurezhangwei/NABA-Atlas/blob/main/parcellate/wm_append_clusters_to_anatomical_tracts_naba.py",
        "hemispheric": NABA_HEMISPHERIC,
        # NABA's seven CPC clusters are all 'h' in the released atlas. The
        # helper script's commissural CPC list conflicts with those data.
        "commissural": CALLOSAL + ("T_MCP",),
    },
}


def atlas_definition(family):
    if family not in ATLAS_DEFINITIONS:
        raise ValueError("WMA atlas must be one of: " + ", ".join(ATLAS_CHOICES))
    return ATLAS_DEFINITIONS[family]


def detect_atlas_family(root, requested="auto"):
    root = Path(root).expanduser()
    if requested != "auto":
        atlas_definition(requested)
        return requested
    detected = [name for name, item in ATLAS_DEFINITIONS.items()
                if all((root / child).is_dir() for child in item["directories"])]
    if len(detected) > 1:
        raise ValueError("Multiple WMA atlases found; select --wma-atlas explicitly")
    return detected[0] if detected else "org"


def tract_groups(family, cluster_dir):
    """Use the published hemisphere categories, retaining actual MRML spelling."""
    definition = atlas_definition(family)
    if family == "org":
        raise ValueError("ORG grouping stays with the unmodified WMA implementation")
    available = {}
    for path in Path(cluster_dir).glob("T*.mrml"):
        key = path.stem.casefold()
        if key in available:
            raise ValueError("Ambiguous atlas MRML name: " + path.name)
        available[key] = path.stem
    missing = [name for name in (*definition["hemispheric"], *definition["commissural"])
               if name.casefold() not in available]
    if missing:
        raise ValueError("Incomplete " + family + " tract definitions: " + ", ".join(missing))
    return {key: [available[name.casefold()] for name in definition[key]]
            for key in ("hemispheric", "commissural")}


def atlas_output_contract(family, cluster_dir):
    groups = tract_groups(family, cluster_dir)
    expected = sorted([name + "_" + side + ".vtp" for name in groups["hemispheric"]
                       for side in ("left", "right")] + [name + ".vtp" for name in groups["commissural"]])
    return {"atlas_family": family, "expected_tract_files": expected,
            "minimum_anatomical_tract_count": len(expected),
            "minimum_nonempty_anatomical_tract_count": (3 * len(expected) + 3) // 4,
            "groups": groups}


def validate_group_hemispheres(cluster_dir, groups):
    """Reject a bundle inventory that disagrees with its reference cluster labels."""
    cluster_dir = Path(cluster_dir)
    rows = (cluster_dir / "cluster_hemisphere_location.txt").read_text(encoding="utf-8").splitlines()[1:]
    labels = {}
    for row in rows:
        if not row.strip():
            continue
        key, value = row.split()
        if key in labels or value not in {"h", "c"}:
            raise ValueError("Invalid atlas hemisphere assignment: " + row)
        labels[key] = value
    if set(labels) != {f"cluster_{index:05d}.vtp" for index in range(1, 801)}:
        raise ValueError("Atlas hemisphere file must assign all 800 reference clusters")
    for kind, expected in (("hemispheric", "h"), ("commissural", "c")):
        for name in groups[kind]:
            text = (cluster_dir / (name + ".mrml")).read_text(encoding="utf-8")
            clusters = set(re.findall(r"cluster_[0-9]{5}\.vtp", text))
            if not clusters or any(labels.get(cluster) != expected for cluster in clusters):
                raise ValueError("Atlas tract/hemisphere mismatch: " + name)
    return {"status": "passed", "reference_clusters": len(labels),
            "tract_definitions": sum(len(group) for group in groups.values())}


def parameterized_author_shell(source, registration_name, clustering_name):
    """Change only the two atlas paths in WMA's installed BSD-licensed driver."""
    replacements = {
        "RegAtlasFolder=$AtlasBaseFolder/ORG-RegAtlas-100HCP": 'RegAtlasFolder="$AtlasBaseFolder/' + registration_name + '"',
        "FCAtlasFolder=$AtlasBaseFolder/ORG-800FC-100HCP": 'FCAtlasFolder="$AtlasBaseFolder/' + clustering_name + '"',
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError("Unrecognized WMA driver layout; update the managed WMA runtime")
        source = source.replace(old, new)
    return source


def atlas_hemisphere_source(source):
    """Respect the new atlas labels instead of WMA's five ORG CPC overrides."""
    pattern = r"(?m)^([ \t]*)location_data\[(145|159|557|677|770)\]\[1\] = b'h'[ \t]*$"
    matches = list(re.finditer(pattern, source))
    if not matches:
        if re.search(r"location_data\[[0-9]+\]\[1\]\s*=", source):
            raise ValueError("Unrecognized atlas-specific hemisphere overrides in WMA")
        return source
    if len(matches) != 5 or {match[2] for match in matches} != {'145', '159', '557', '677', '770'}:
        raise ValueError("Unrecognized ORG hemisphere overrides in WMA")
    return re.sub(pattern, r"\1pass  # Use this atlas's published hemisphere assignments.", source)
