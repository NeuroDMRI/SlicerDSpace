#!/usr/bin/env python3
"""Prepare everything the selected targets need, with one licence confirmation (run with PythonSlicer).

Without --execute the script only lists the plan: every component with its download size, licence name,
licence link and local licence file. With --execute it installs. --accept-licences accepts all listed licences.
"""

import argparse
import json
import os
import sys
from dataclasses import fields
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def build_cli_plan(targets, user_config, *, doctor=None, wsl_distributions=None, system=None, machine=None):
    """Closure, doctor report and setup plan for the targets, from the saved user configuration."""
    from shared.environment import EnvironmentDoctorConfig, run_environment_doctor
    from shared.environment_check import config_from_dict
    from scripts.setup_python_packages import dcm2niix_pending
    from shared.selected_setup import build_setup_plan, need_flags
    from shared.target_closure import expand_closure

    from shared.algorithm_selection import BUNDLES, TRACKING, algorithms, choice_value

    defaults, deps = user_config["run_defaults"], dict(user_config["external_dependencies"])
    stages = set(defaults.get("microstructure_stages") or ())
    # S-B5: saved selections may be lists ("csd_act,ukf") or "all"; the GUI parses them with the same function.
    closure = expand_closure(targets, tractography_algorithms=algorithms(defaults.get("tractography_backend"), TRACKING, "csd_act"),
                             bundle_backends=algorithms(defaults.get("bundle_backend"), BUNDLES, "wmql"),
                             microstructure_free_water="freewater" in stages, microstructure_noddi="noddi" in stages,
                             microstructure_dki="dki" in stages)
    known = {field.name for field in fields(EnvironmentDoctorConfig)}
    values = {key: defaults[key] for key in ("preprocessing_backend", "brain_mask_backend", "brain_mask_device", "pnl_mask_route",
                                             "ddseg_device", "surface_device", "tractcloud_device",
                                             "tractography_backend", "connectome_backend", "wma_atlas",
                                             "dki_fit_method") if defaults.get(key)}
    values.update({key: value for key, value in deps.items() if key in known})
    values.update(targets=list(closure.targets), repo_root=str(REPO_ROOT),
                  run_free_water="freewater" in stages, run_noddi="noddi" in stages, run_dki="dki" in stages)
    if not deps.get("python_executable") and not deps.get("python_environment"):
        values["python_executable"] = sys.executable  # S-B4: a saved interpreter is kept; the current one only as the default
    if "tractography" in closure.targets:  # S-B6: the doctor checks the closed algorithm set, as the GUI does
        values["tractography_backend"] = choice_value(closure.algorithms_text)
    report = (doctor or run_environment_doctor)(config_from_dict(values))
    plan = build_setup_plan(
        report, **need_flags(closure.targets, closure.tractography_algorithms),
        brain_mask_backend=defaults.get("brain_mask_backend") or "pnl_cnn", pnl_mask_route=defaults.get("pnl_mask_route") or "auto",
        wma_atlas=defaults.get("wma_atlas") or "org",
        backend_paths=deps, closure_targets=closure.targets, system=system, machine=machine,
        wsl_distributions=wsl_distributions, slicer_executable=deps.get("slicer_executable"),
        dcm2niix_missing=dcm2niix_pending())  # S-B3: this script runs in Slicer's Python (PythonSlicer)
    plan["closure_targets"] = list(closure.targets)
    return closure, report, plan


def describe_components(components, pending):
    """Plain text lines: component, size, licence name, link and local licence file."""
    from scripts.download_model_assets import _format_bytes

    lines = []
    for item in components:
        size = _format_bytes(item["size_bytes"]) if item["size_bytes"] else "size not known"
        lines.append("- %s (%s)" % (item["title"], size))
        licence = item["licence"]
        if licence:
            lines.append("    licence: %s%s" % (licence["name"], "" if licence["id"] in pending else " (already accepted)"))
            lines.append("    link: " + licence["url"])
            lines.append("    local file: " + (licence["file"] or "none"))
            for package in licence.get("packages", ()):  # S3-2: each package with its licence, link and local file
                lines.append("      %s: %s | %s | %s" % (package["name"], package["license"], package["url"],
                                                          package["file"] or "no local file"))
    return lines


def external_environment(program, environment=None):
    """Environment overrides for a command: an external interpreter gets Slicer's Python variables removed (S-B7).

    Same rule as the GUI self-check: only a Python that is not the running one needs the clean environment.
    """
    import os

    environment = dict(environment or {})
    if Path(str(program)).absolute() != Path(sys.executable).absolute():
        from shared.external_process import external_python_environment

        clean = external_python_environment(path_prepend=[Path(program).parent])
        environment.update({key: None for key in os.environ if key not in clean})
        environment.update(clean)
    return environment


def self_check_commands(folder, user_config, closure_targets, step, *, doctor=None):
    """Build the self-check commands from a fresh doctor report, after the other steps changed the setup."""
    from shared.setup_selftest import requests_for_report
    from shared.managed_runtimes import write_json

    _closure, report, _plan = build_cli_plan(closure_targets, user_config, doctor=doctor)
    deps = user_config["external_dependencies"]
    requests = requests_for_report(report, asset_root=deps.get("model_assets_dir"), backend_paths=deps,
                                   device_choices=user_config["run_defaults"], repo_root=REPO_ROOT)
    commands, receipts = [], []
    for index, request in enumerate(requests):
        path, receipt = Path(folder) / ("functional-%d-request.json" % index), Path(folder) / ("functional-%d-result.json" % index)
        write_json(path, request)
        program = request.get("python") or sys.executable
        commands.append((program, ["-u", step["script"], "--request", str(path), "--result", str(receipt)],
                         external_environment(program, {"MPLBACKEND": "Agg"})))
        receipts.append(receipt)
    return commands, receipts


def running_slicer_pythons():
    """Paths that name the Slicer running this script: its interpreter, its PythonSlicer launcher and python-real (S3-1).

    W-B-CARRY-1: exact paths only. A virtualenv whose ``bin/python`` links to python-real is an external interpreter.
    """
    from shared.external_process import find_pythonslicer, interpreter_path

    current = interpreter_path(sys.executable)
    found = {current}
    for folder in current.parents[:4]:
        launcher = find_pythonslicer(folder)
        if launcher:
            found.add(interpreter_path(launcher))
    for name in ("python-real", "python-real.exe"):
        if (current.parent / name).is_file():
            found.add(current.parent / name)
    return found


def is_running_slicer_python(selected):
    """True when ``selected`` is this Slicer's own Python (PythonSlicer or python-real), like the GUI's classification."""
    from shared.external_process import interpreter_path

    return interpreter_path(selected) in running_slicer_pythons()


def make_python_command(configured):
    """python_command for setup_steps: a request without its own interpreter uses the configured one, like the GUI (S-B7).

    S3-1: a saved interpreter that is this Slicer's own Python (the Windows ps1 saves its PythonSlicer path) runs
    directly as Slicer's Python, so protect_bundled stays true and torch goes through the PyTorch-extension installer.
    """
    def python_command(override):
        selected = str(override or configured or sys.executable)
        return (sys.executable if is_running_slicer_python(selected) else selected), []

    return python_command


def main(argv=None, *, runner=None, doctor=None, wsl_distributions=None, system=None, machine=None, log=print):
    from shared.selected_setup import run_setup_steps, setup_steps
    from shared.setup_components import accept_licences, pending_licences, plan_components, required_licences
    from shared.user_config import default_user_config_path, load_user_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", required=True, help="Comma- or space-separated target names")
    parser.add_argument("--execute", action="store_true", help="Install; without it only the plan is listed")
    parser.add_argument("--accept-licences", action="store_true", help="Accept every listed licence")
    args = parser.parse_args(argv)
    targets = [token for part in args.targets.split() for token in part.split(",") if token]
    config = load_user_config()
    closure, report, plan = build_cli_plan(targets, config, doctor=doctor, wsl_distributions=wsl_distributions,
                                            system=system, machine=machine)
    components = plan_components(plan)
    required = required_licences(components)
    pending = pending_licences(required, config["accepted_licences"])
    log("Targets: " + ", ".join(closure.targets))
    log("Components:" if components else "Nothing to install.")
    for line in describe_components(components, pending):
        log(line)
    for key in ("extensions", "restart_extensions"):
        if plan.get(key):  # SPEC-GAP: the CLI cannot install Slicer extensions; it lists them
            log("Install in Slicer, then restart (%s): %s" % (key.replace("_", " "), ", ".join(plan[key])))
    for item in plan["manual_items"]:
        log("Still pending: %s. %s" % (item["label"], item["hint"]))
    if not args.execute:
        return 0
    if pending:
        if not args.accept_licences:
            log("Licences not accepted: %s. Run again with --accept-licences to accept all of them." % ", ".join(pending))
            return 2
        accept_licences(pending)
        config = load_user_config()
    folder = default_user_config_path().parent / "setup-history" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    folder.mkdir(parents=True, exist_ok=True)
    configured = getattr(report, "python_executable", None)

    python_command = make_python_command(configured)

    steps = setup_steps(plan, config["accepted_licences"], folder=folder, config=config["external_dependencies"],
                        python_command=python_command)
    records = run_setup_steps(
        steps, runner=runner, log=log,
        self_check=lambda step: self_check_commands(folder, load_user_config(), closure.targets, step, doctor=doctor))
    (folder / "result.json").write_text(json.dumps({"steps": records}, indent=2) + "\n")
    log(json.dumps({"steps": records}))
    return 0 if records and all(record["status"] == "completed" for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
