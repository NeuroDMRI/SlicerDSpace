#!/usr/bin/env python3
"""Apply a supported atlas through WMA, preserving its registration and clustering."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.bundled_python import activate_bundled_python

activate_bundled_python()

from tractography.wma_atlases import (atlas_definition, atlas_output_contract,
                                      parameterized_author_shell, atlas_hemisphere_source,
                                      validate_group_hemispheres)


def append_with_atlas_groups(author_script, groups, arguments):
    """Keep the author's append/attribute code and replace only its ORG name lists."""
    previous = sys.argv
    try:
        namespace = runpy.run_path(str(author_script), run_name="_wma_atlas_groups")
        main = namespace.get("main")
        if not callable(main) or not all(callable(namespace.get(name)) for name in (
                "get_hemispheric_mono_bundle_names", "get_commissural_bundle_names")):
            raise ValueError("This WMA append implementation cannot accept atlas-specific tract groups")
        main.__globals__["get_hemispheric_mono_bundle_names"] = lambda _: list(groups["hemispheric"])
        main.__globals__["get_commissural_bundle_names"] = lambda _: list(groups["commissural"])
        sys.argv = [str(author_script), *arguments]
        main()
    finally:
        sys.argv = previous


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("east-west", "naba"))
    parser.add_argument("--author-shell", type=Path)
    parser.add_argument("--append-author", type=Path)
    parser.add_argument("--groups", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    if args.append_author:
        append_with_atlas_groups(args.append_author, json.loads(args.groups.read_text()), arguments)
        return 0
    if not args.family or not args.author_shell:
        parser.error("--family and --author-shell are required")
    definition = atlas_definition(args.family)
    root = Path(arguments[arguments.index("-a") + 1])
    reg_name, cluster_name = definition["directories"]
    contract = atlas_output_contract(args.family, root / cluster_name)
    validate_group_hemispheres(root / cluster_name, contract["groups"])
    source = args.author_shell.read_text(encoding="utf-8")
    author_append = args.author_shell.with_name("wm_append_clusters_to_anatomical_tracts.py")
    with tempfile.TemporaryDirectory(prefix="slicerdspace-wma-atlas-") as temporary:
        temp = Path(temporary)
        driver = temp / "apply_atlas.sh"
        driver.write_text(parameterized_author_shell(source, reg_name, cluster_name), encoding="utf-8")
        groups = temp / "groups.json"
        groups.write_text(json.dumps(contract["groups"]), encoding="utf-8")
        shim = temp / author_append.name
        # Resolve through the supplied WMA/Slicer environment, not a new runtime.
        shim.write_text("#!/bin/sh\nexec python " + shlex.quote(str(Path(__file__).resolve()))
                        + " --append-author " + shlex.quote(str(author_append))
                        + " --groups " + shlex.quote(str(groups)) + ' -- "$@"\n', encoding="utf-8")
        shim.chmod(0o755)
        assessment_name = "wm_assess_cluster_location_by_hemisphere.py"
        author_assessment = args.author_shell.with_name(assessment_name)
        assessment = atlas_hemisphere_source(author_assessment.read_text(encoding="utf-8"))
        (temp / "assessment.py").write_text(assessment, encoding="utf-8")
        assessment_shim = temp / assessment_name
        assessment_shim.write_text("#!/bin/sh\nexec python " + shlex.quote(str(temp / "assessment.py"))
                                   + ' "$@"\n', encoding="utf-8")
        assessment_shim.chmod(0o755)
        env = dict(os.environ, PATH=str(temp) + os.pathsep + os.environ.get("PATH", ""))
        print(json.dumps({"atlas_family": args.family, "author_shell_sha256": hashlib.sha256(source.encode()).hexdigest(),
                          "tract_groups": contract["groups"]}), flush=True)
        return subprocess.call(["bash", str(driver), *arguments], env=env)


if __name__ == "__main__":
    raise SystemExit(main())
