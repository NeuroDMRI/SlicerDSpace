"""Run tracking variants separately while sharing verified anatomical products."""

from dataclasses import replace
import json
import time
from datetime import datetime, timezone

from shared.algorithm_selection import algorithms

SHARED = {"validation", "preprocessing", "segmentation", "parcellation", "surface", "surface_postprocess",
          "microstructure", "microstructure_freewater", "microstructure_noddi", "microstructure_dki",
          "microstructure_ifw"}


def variants(config):
    from shared.io_layout import normalize_run_name
    selected = algorithms(config.tractography_backend)
    for index, backend in enumerate(selected):
        # WMA consumes only a UKF tractogram (closure rule C1): it runs in the UKF variant.
        wma_only_on_ukf = "ukf" in selected and not config.bundle_tractography_backend
        targets = tuple(target for target in config.targets if
                        # The SAF target uses the CSD FOD; without the SAF algorithm it runs once, in the first variant.
                        (backend == "saf" or target != "saf" or ("saf" not in selected and index == 0))
                        and (index == 0 or target not in SHARED)
                        and not (target == "wma" and wma_only_on_ukf and backend != "ukf")
                        and (not config.bundle_tractography_backend or backend == config.bundle_tractography_backend
                             or target not in {'bundles', 'wma', 'tractcloud'}))
        requesters = config.closure_algorithm_requesters.get(backend)
        if requesters and index > 0:
            # An algorithm the closure added (for example UKF for WMA) runs only for the targets that need it.
            targets = tuple(target for target in targets if target in {"tractography", *requesters})
        if not targets:
            continue
        yield replace(config, tractography_backend=backend, closure_applied=True, write_run_record=False,
                      preflight_blocked_targets=config.preflight_blocked_by_backend.get(
                          backend, config.preflight_blocked_targets),
                      bundle_tractography_backend=(config.bundle_tractography_backend
                                                   if backend == config.bundle_tractography_backend else None),
                      run_name=normalize_run_name(config.run_name) + "__" + backend, targets=targets,
                      microstructure_run_free_water=config.microstructure_run_free_water if index == 0 else False,
                      microstructure_run_noddi=config.microstructure_run_noddi if index == 0 else False,
                      microstructure_run_dki=config.microstructure_run_dki if index == 0 else False,
                      microstructure_run_ifw=config.microstructure_run_ifw if index == 0 else False)


def plan(config, *, write=True):
    from shared.orchestrator import RunPlanResult, plan_subject_products, _run_plan_path, _write_run_plan, _with_closure
    config = _with_closure(config)
    from shared.io_layout import subject_layout, normalize_run_name
    children = [plan_subject_products(child, write=write) for child in variants(config)]
    pending_shared = {item.target for item in children[0].items if item.target in SHARED and item.status in {'will_run', 'skipped'}}
    items = []
    for index, (child, result) in enumerate(zip(variants(config), children)):
        for item in result.items:
            item = replace(item, task_name=item.task_name + ("@" + child.tractography_backend if item.target not in SHARED else ""),
                           reason=f"[{child.tractography_backend}] {item.reason}")
            missing = {key for key, value in item.dependencies.items() if value in {None, 'missing'}}
            if index and item.status == 'blocked' and item.reason_category == 'upstream' and missing and missing <= pending_shared:
                item = replace(item, status='will_run', reason=f"[{child.tractography_backend}] Waiting for shared products from the first variant",
                               runnable=True, will_run=True, reason_category='upstream_pending')
            items.append(item)
    layout = subject_layout(config.subject_root, output_root=config.output_root)
    result = RunPlanResult(config.subject_id, config.subject_root.resolve(), layout.outputs,
                           normalize_run_name(config.run_name), tuple(dict.fromkeys(item.target for item in items)),
                           _run_plan_path(layout, normalize_run_name(config.run_name)), tuple(items))
    if write:
        _write_run_plan(result)
    return result


def run(config, *, log=None):
    from shared.orchestrator import (ProductOrchestratorResult, ProductTaskRecord, run_subject_products,
                                     _archive_existing_run_records, _normalize_bundle_backends, _with_closure)
    config = _with_closure(config)
    from shared.io_layout import subject_layout, normalize_run_name
    layout = subject_layout(config.subject_root, output_root=config.output_root)
    name = normalize_run_name(config.run_name)
    _archive_existing_run_records(layout, name)
    planned = plan(config)
    destination = layout.run_metadata / f"{name}_orchestration_manifest.json"
    tasks, children = [], []
    started = time.monotonic()
    payload = dict(kind="slicerddsurfer_product_orchestration", version=1, subject_id=config.subject_id,
                   subject_root=str(config.subject_root.resolve()), output_root=str(layout.outputs), run_name=name,
                   targets=list(planned.targets), tractography_backends=list(algorithms(config.tractography_backend)),
                   bundle_tractography_backend=config.bundle_tractography_backend,
                   bundle_backends=list(_normalize_bundle_backends(config.bundle_backend)),
                   closure={"required_by": {key: list(value) for key, value in config.target_required_by.items()}},
                   preflight_blocked_by_backend=config.preflight_blocked_by_backend,
                   run_plan_path=str(planned.run_plan_path), status="running", tasks=tasks, child_runs=children,
                   execute=config.execute, started_at=datetime.now(timezone.utc).isoformat(),
                   note="Tracking variants run sequentially; shared products run once. Bundle/connectome outputs are variant-specific; QC gates are unchanged.")

    def publish():
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n")
        temporary.replace(destination)

    publish()
    for child in variants(config):
        child_path = layout.run_metadata / f"{child.run_name}_orchestration_manifest.json"
        entry = dict(backend=child.tractography_backend, manifest_path=str(child_path), status='running',
                     started_at=datetime.now(timezone.utc).isoformat())
        children.append(entry)
        publish()
        if log and config.execute:  # plan-only runs keep stdout as pure JSON
            log("[Algorithms] Starting tracking variant: " + child.tractography_backend)
        try:
            result = run_subject_products(child, log=log)
            record = json.loads(result.run_manifest_path.read_text())
            entry.update(manifest_path=str(result.run_manifest_path), status=record.get('status', 'finished'))
            for task in record.get("tasks", []):
                shared = task["name"].split('.', 1)[0] in SHARED or task["name"].startswith(("microstructure.", "surface."))
                tasks.append(dict(task, name=task["name"] + ("" if shared else "@" + child.tractography_backend),
                                  product=task["name"], tracking_backend=child.tractography_backend))
        except Exception as exc:
            entry['status'] = 'failed'
            try:
                failed = json.loads(child_path.read_text())
                for task in failed.get('tasks', []):
                    shared = task['name'].split('.', 1)[0] in SHARED or task['name'].startswith(('microstructure.', 'surface.'))
                    tasks.append(dict(task, name=task['name'] + ('' if shared else '@' + child.tractography_backend),
                                      product=task['name'], tracking_backend=child.tractography_backend))
            except (OSError, ValueError, KeyError, TypeError):
                pass
            tasks.append(dict(name="execution.failure@" + child.tractography_backend,
                              status="failed", reason=str(exc), outputs={}))
        publish()
    states = {task.get("status") for task in tasks}
    failed = bool(states & {"failed", "partial_failed", "blocked", "error"}) or any(
        child['status'] in {'failed', 'partial_failed', 'completed_with_blocks'} for child in children)
    payload["status"] = "failed" if failed else "completed" if config.execute else "planned"
    payload["elapsed_seconds"] = time.monotonic() - started
    payload['ended_at'] = datetime.now(timezone.utc).isoformat()
    publish()
    if config.execute:
        # T-C5: one run record from the final aggregate manifest, not one per child.
        from shared.orchestrator import _write_run_record_best_effort
        _write_run_record_best_effort(destination, log)
    records = tuple(ProductTaskRecord(name=task["name"], status=task["status"], reason=task.get("reason", "")) for task in tasks)
    return ProductOrchestratorResult(config.subject_id, config.subject_root.resolve(), destination, records, planned.run_plan_path)
