#!/usr/bin/env python3
"""Batch runner for SlicerDSpace subject-root workflows."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.external_process import text_subprocess_kwargs
from shared.pipeline import RAW_INPUT_FILES, inspect_subject_directory, is_raw_subject_root
from shared.io_layout import subject_layout, normalize_run_name


@dataclass(frozen=True)
class BatchSubject:
    subject_id: str
    subject_root: Path
    source: str
    mode: str


@dataclass(frozen=True)
class SubjectRunResult:
    subject_id: str
    subject_root: Path
    command: list[str]
    returncode: int
    stdout_log: Path
    stderr_log: Path
    elapsed_seconds: float
    run_status: str = 'unknown'
    run_manifest_path: Path | None = None
    task_counts: dict[str, int] | None = None
    status_source: str = 'unavailable'
    manifest_error: str | None = None


def _parse_jobs(value: str) -> int:
    jobs = int(value)
    if jobs < 1:
        raise argparse.ArgumentTypeError("batch size/jobs must be at least 1")
    return jobs


def _unique_path_candidates(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def _direct_raw_files(path: Path) -> bool:
    return path.is_dir() and all((path / filename).exists() for filename in RAW_INPUT_FILES.values())


def _arrange_subject_input(subject_dir: Path, mode: str) -> bool:
    if mode == "none" or is_raw_subject_root(subject_dir) or not _direct_raw_files(subject_dir):
        return False
    input_dir = subject_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    for filename in RAW_INPUT_FILES.values():
        source = subject_dir / filename
        target = input_dir / filename
        if target.exists():
            continue
        if mode == "copy":
            shutil.copy2(source, target)
        elif mode == "symlink":
            target.symlink_to(source)
        else:
            raise ValueError(f"Unsupported arrange mode: {mode}")
    return True


def _parse_subject_list(path: Path, batch_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for raw_line in path.expanduser().read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.replace(",", " ").split() if part.strip()]
        if not parts:
            continue
        value = parts[-1]
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = batch_root / candidate
        candidates.append(candidate)
    return candidates


def discover_subjects(
    batch_root: Path,
    *,
    subject_glob: str = "*",
    subject_list: Path | None = None,
    arrange_inputs: str = "none",
    include_missing: bool = False,
) -> tuple[list[BatchSubject], list[dict[str, str]]]:
    root = batch_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Batch root does not exist or is not a directory: {root}")

    if subject_list:
        candidates = _parse_subject_list(subject_list, root)
    else:
        candidates = [path for path in sorted(root.glob(subject_glob)) if path.is_dir()]
    candidates = _unique_path_candidates(candidates)

    subjects: list[BatchSubject] = []
    skipped: list[dict[str, str]] = []
    for candidate in candidates:
        arranged = _arrange_subject_input(candidate, arrange_inputs)
        subject_id = candidate.name
        inspected = inspect_subject_directory(candidate, subject_id)
        if inspected["mode"] == "missing":
            inspected = inspect_subject_directory(root, subject_id)
        if inspected["mode"] == "missing":
            skipped.append(
                {
                    "path": candidate.as_posix(),
                    "subject_id": subject_id,
                    "reason": "missing required input/data.nii.gz, bvals, or bvecs",
                }
            )
            if include_missing:
                subjects.append(BatchSubject(subject_id, candidate, "subject-list" if subject_list else "glob", "missing"))
            continue
        subject_root = Path(inspected["subject_root"] or inspected["subject_dir"]).expanduser().resolve()
        mode = str(inspected["mode"])
        if arranged:
            mode = f"{mode}+arranged"
        subjects.append(BatchSubject(subject_id, subject_root, "subject-list" if subject_list else "glob", mode))
    return subjects, skipped


def build_subject_command(
    subject: BatchSubject,
    *,
    pipeline_script: Path,
    python_executable: Path,
    output_root: Path | None,
    run_name: str,
    execute: bool,
    forwarded_args: list[str],
) -> list[str]:
    command = [
        str(python_executable.expanduser()),
        str(pipeline_script.expanduser()),
        "--subject-root",
        subject.subject_root.as_posix(),
        "--subject-id",
        subject.subject_id,
        "--run-name",
        run_name,
    ]
    if output_root is not None:
        command.extend(["--output-root", (output_root.expanduser().resolve() / subject.subject_id).as_posix()])
    if execute:
        command.append("--execute")
    command.extend(forwarded_args)
    return command


def _command_option(command: list[str], option: str, default: str) -> str:
    value = default
    for index, argument in enumerate(command):
        if argument.startswith(option + '='):
            value = argument.split('=', 1)[1]
        elif argument == option and index + 1 < len(command):
            value = command[index + 1]
    return value


def _manifest_identity(path: Path):
    try:
        stat = path.stat()
        return stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
    except OSError:
        return None


def _run_subject(command: list[str], subject: BatchSubject, log_dir: Path, *, environment=None) -> SubjectRunResult:
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in subject.subject_id)
    stdout_log = log_dir / f"{safe_id}.stdout.log"
    stderr_log = log_dir / f"{safe_id}.stderr.log"
    output = _command_option(command, '--output-root', '')
    run_name = normalize_run_name(_command_option(command, '--run-name', 'Test1'))
    child_manifest = subject_layout(subject.subject_root, output_root=Path(output) if output else None).run_metadata / f'{run_name}_orchestration_manifest.json'
    old_identity = _manifest_identity(child_manifest)
    started = time.monotonic()
    # Stream independent logs so a long cohort does not retain every process's
    # output in the batch driver's memory, and logs are readable during a run.
    with stdout_log.open("w", encoding="utf-8") as stdout, stderr_log.open("w", encoding="utf-8") as stderr:
        try:
            options = text_subprocess_kwargs()
            if environment is not None:
                options['env'] = {**os.environ, **environment, 'SLICERDSPACE_COHORT_SUBJECT': subject.subject_id}
            completed = subprocess.run(command, stdout=stdout, stderr=stderr, **options)
        except OSError as exc:
            stderr.write(f"Could not start subject process: {exc}\n")
            completed = subprocess.CompletedProcess(command, 127)
    process_elapsed_seconds = round(time.monotonic() - started, 3)
    task_counts = None
    manifest_error = None
    run_status = 'failed' if completed.returncode else 'unknown'
    status_source = 'process_exit_code'
    manifest_path = None
    try:
        if old_identity == _manifest_identity(child_manifest):
            raise ValueError('No fresh child orchestration manifest was written.')
        record = json.loads(child_manifest.read_text(encoding='utf-8'))
        if record.get('subject_id') != subject.subject_id or record.get('run_name') != run_name:
            raise ValueError('Child manifest subject/run identity does not match this batch entry.')
        terminal = {'completed', 'completed_with_blocks', 'failed', 'cancelled', 'planned'}
        if record.get('status') not in terminal:
            raise ValueError('Child manifest does not contain a terminal run status.')
        tasks = record.get('tasks')
        if not isinstance(tasks, list) or any(not isinstance(task, dict) or not isinstance(task.get('status'), str) for task in tasks):
            raise ValueError('Child manifest has invalid task records.')
        task_counts = {}
        for task in tasks:
            state = task['status']
            task_counts[state] = task_counts.get(state, 0) + 1
        run_status = record['status']
        if any(task_counts.get(state) for state in ('failed', 'partial_failed')):
            run_status = 'failed'
        elif task_counts.get('cancelled'):
            run_status = 'cancelled'
        elif task_counts.get('blocked'):
            run_status = 'completed_with_blocks'
        elif any(task_counts.get(state) for state in ('running', 'pending', 'will_run')) or (
                '--execute' in command and task_counts.get('planned')):
            run_status = 'unknown'
        if completed.returncode:
            run_status = 'cancelled' if run_status == 'cancelled' else 'failed'
        manifest_path = child_manifest
        status_source = 'child_manifest'
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        manifest_error = str(exc)
        # Custom runners may not implement our manifest contract. Keep that
        # distinction explicit instead of treating stale manifests as evidence.
        builtins = {(REPO_ROOT / 'shared/run_subject_pipeline_cli.py').resolve(), (REPO_ROOT / 'run_subject_pipeline.py').resolve()}
        if len(command) > 1 and Path(command[1]).resolve() not in builtins:
            run_status = 'failed' if completed.returncode else 'process_completed'
    return SubjectRunResult(
        subject_id=subject.subject_id,
        subject_root=subject.subject_root,
        command=command,
        returncode=completed.returncode,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        elapsed_seconds=process_elapsed_seconds,
        run_status=run_status,
        run_manifest_path=manifest_path,
        task_counts=task_counts,
        status_source=status_source,
        manifest_error=manifest_error,
    )


def _write_manifest(path: Path, payload: dict[str, object]) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _resource_fraction(value):
    try:
        number = float(value[:-1]) / 100 if value.endswith('%') else float(value)
        if not 0 < number <= 1:
            raise ValueError()
        return number
    except ValueError:
        raise argparse.ArgumentTypeError('Use a fraction in (0,1], or a percentage such as 80%')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run SlicerDSpace over a directory of subject roots. Unknown options are forwarded "
            "unchanged to every subject CLI, for example --all, --targets, --brain-mask-backend, "
            "--pnl-mask-python, --pnl-mask-script, --pnl-mask-model-dir, --pnl-mask-ants-bin-dir, "
            "--postprocess-freesurfer-home, or other external paths."
            " Microstructure defaults to DTI only; --microstructure-stages, --run-free-water, "
            "--run-noddi, --run-dki, --dki-fit-method, --dki-matlab-executable and --dki-mkcurve-root "
            "are forwarded unchanged."
        )
    )
    parser.add_argument("--batch-root", required=True, type=Path, help="Directory containing one subdirectory per subject.")
    parser.add_argument("--subject-glob", default="*", help="Glob used under --batch-root when --subject-list is omitted.")
    parser.add_argument("--subject-list", type=Path, help="Optional text file listing subject directories or relative IDs.")
    parser.add_argument('--subject-limit', type=_parse_jobs, help='Process the first N eligible subjects; default: all.')
    parser.add_argument("--output-root", type=Path, help="Batch output root. Each subject writes under <output-root>/<subject-id>.")
    parser.add_argument("--run-name", default="Test1", help="Run name forwarded to each subject. Defaults to Test1.")
    parser.add_argument("--jobs", "--batch-size", dest="jobs", type=_parse_jobs,
                        help="Concurrent subjects; automatic in resource mode, otherwise 1.")
    parser.add_argument('--scheduler', choices=('subjects', 'resources', 'adaptive'), default='subjects',
                        help='adaptive adds CPU-first fair backfilling and resource feedback; resources keeps first-fit admission.')
    parser.add_argument('--resource-budget', type=_resource_fraction, default=.8,
                        help='Adaptive automatic CPU/RAM budget fraction (default: 80%%); explicit budgets override it.')
    parser.add_argument('--resource-history', type=Path, help='Adaptive resource profile file; defaults under batch output metadata.')
    parser.add_argument('--compute-profile', type=Path, help='Validated thread/worker calibration profile; applied only to selected automatic stages.')
    parser.add_argument('--autotune', action='store_true', help='Calibrate selected supported CPU models once before adaptive execution; reuse matching profiles.')
    parser.add_argument('--tuning-max-cpus', type=_parse_jobs, default=4, help='Bounded CPU ceiling per calibration candidate (default: 4).')
    parser.add_argument('--batch-cpus', type=_parse_jobs, help='Declared stage CPU budget; adaptive defaults to --resource-budget of available CPUs.')
    parser.add_argument('--batch-memory-gb', type=float, help='Declared RAM budget in GiB; adaptive uses --resource-budget, resources uses 70%% of available RAM.')
    parser.add_argument('--batch-stage-cpus', type=_parse_jobs, help='Override automatic per-stage CPU reservations.')
    parser.add_argument('--batch-stage-memory-gb', type=float, help='Override header/model-based stage RAM estimates in GiB.')
    parser.add_argument('--batch-gpu-slots', type=_parse_jobs, default=1,
                        help='Global GPU-capable stage slots, not GPU assignment or VRAM enforcement; default: 1.')
    parser.add_argument("--execute", action="store_true", help="Actually execute each subject. Omit for dry-run planning.")
    parser.add_argument(
        "--arrange-inputs",
        choices=("none", "symlink", "copy"),
        default="none",
        help="Optionally create subject/input from raw files placed directly in each subject directory.",
    )
    parser.add_argument("--include-missing", action="store_true", help="Include missing subjects in the manifest instead of only skipped.")
    parser.add_argument("--continue-on-error", action="store_true", help="Keep launching remaining subjects after failures. Otherwise, finish and record active subjects, then stop.")
    parser.add_argument('--allow-partial-success', action='store_true', help='Allow exit code 0 when child runs are blocked; the batch still records completed_with_blocks.')
    parser.add_argument("--manifest", type=Path, help="Batch manifest path. Defaults under output root or batch root.")
    parser.add_argument("--log-dir", type=Path, help="Per-subject stdout/stderr log directory.")
    parser.add_argument(
        "--pipeline-script",
        type=Path,
        default=REPO_ROOT / "shared" / "run_subject_pipeline_cli.py",
        help="Subject-level CLI script to execute.",
    )
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable), help="Python executable for subject CLI.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, forwarded_args = parser.parse_known_args(argv)
    resource_mode = args.scheduler in ('resources', 'adaptive')
    if args.autotune and args.scheduler != 'adaptive':
        parser.error('--autotune requires --scheduler adaptive')
    if args.scheduler == 'resources':
        if int(_command_option(forwarded_args, '--max-parallel-models', '1')) != 1:
            parser.error('Resource scheduling currently requires --max-parallel-models 1 to avoid nested reservations')

    try:
        subjects, skipped = discover_subjects(
            args.batch_root,
            subject_glob=args.subject_glob,
            subject_list=args.subject_list,
            arrange_inputs=args.arrange_inputs,
            include_missing=args.include_missing,
        )
    except Exception as exc:
        parser.error(str(exc))

    ids: dict[str, Path] = {}
    for subject in subjects:
        if subject.subject_id in ids:
            parser.error(f"Duplicate subject ID {subject.subject_id!r}: {ids[subject.subject_id]} and {subject.subject_root}")
        ids[subject.subject_id] = subject.subject_root

    output_base = args.output_root.expanduser().resolve() if args.output_root else args.batch_root.expanduser().resolve() / "batch_outputs"
    manifest_path = args.manifest or output_base / "00_batch_metadata" / f"batch_{time.strftime('%Y%m%d_%H%M%S')}.json"
    log_dir = args.log_dir.expanduser().resolve() if args.log_dir else manifest_path.expanduser().resolve().parent / "logs"

    runnable_subjects = [subject for subject in subjects if subject.mode != "missing"]
    if args.subject_limit is not None:
        selected = runnable_subjects[:args.subject_limit]
        skipped.extend(dict(subject_id=subject.subject_id, reason='subject limit')
                       for subject in runnable_subjects[args.subject_limit:])
        runnable_subjects = selected
    resource_plan = None
    if resource_mode:
        from shared.cohort_planning import plan_cohort
        try:
            resource_plan = plan_cohort(runnable_subjects, forwarded_args, jobs=args.jobs,
                cpu_budget=args.batch_cpus, memory_gb=args.batch_memory_gb,
                stage_cpus=args.batch_stage_cpus, stage_memory_gb=args.batch_stage_memory_gb,
                gpu_slots=args.batch_gpu_slots, adaptive=args.scheduler == 'adaptive',
                resource_fraction=args.resource_budget,
                history_path=args.resource_history or output_base / '00_batch_metadata/resource_history.json',
                tuning_profile=args.compute_profile)
        except (ValueError, ImportError) as exc:
            parser.error(str(exc))
        args.jobs = resource_plan['jobs']
        if args.autotune and args.execute:
            profile = args.compute_profile or output_base / '00_batch_metadata/compute_profile.json'
            supported = sorted(set(resource_plan['selected_stages']) & {'microstructure_dki', 'microstructure_mkcurve', 'wma'})
            if 'preprocessing' in resource_plan['selected_stages']:
                print('[Autotune] Calibration needs the final corrected DWI. Retaining declared settings; '
                      'calibrate the preprocessed inputs before the next batch.', flush=True)
                supported = []
            if supported:
                command = [str(args.python_executable), str(REPO_ROOT / 'scripts/autotune_batch.py'),
                    '--output', str(profile), '--stages', ','.join(supported),
                    '--max-cpus', str(min(args.tuning_max_cpus, resource_plan['cpu_budget'])),
                    '--cpu-budget', str(resource_plan['cpu_budget']), '--memory-gb', str(resource_plan['memory_gb']),
                    '--jobs', str(args.jobs)]
                for subject in runnable_subjects[:3]:
                    command += ['--subject-root', str(subject.subject_root)]
                command += ['--', *forwarded_args]
                subprocess.run(command, check=True)
                resource_plan = plan_cohort(runnable_subjects, forwarded_args, jobs=args.jobs,
                    cpu_budget=resource_plan['cpu_budget'], memory_gb=resource_plan['memory_gb'],
                    stage_cpus=args.batch_stage_cpus, stage_memory_gb=args.batch_stage_memory_gb,
                    gpu_slots=args.batch_gpu_slots, adaptive=True, resource_fraction=args.resource_budget,
                    history_path=resource_plan['history_path'], tuning_profile=profile)
        print(f"[Batch] Resource plan: up to {args.jobs} subjects; "
              f"{resource_plan['cpu_budget']} CPU threads, {resource_plan['memory_gb']:.1f} GiB RAM; "
              f"{resource_plan['gpu_slots']} GPU-capable stage slot(s).", flush=True)
        if not _command_option(forwarded_args, '--max-parallel-models', ''):
            forwarded_args += ['--max-parallel-models', '1']
        if not _command_option(forwarded_args, '--microstructure-free-water-n-jobs', ''):
            forwarded_args += ['--microstructure-free-water-n-jobs', '1']
        for flag in ('--surface-cpu-threads', '--mrtrix-nthreads', '--preprocessing-eddy-threads'):
            if not _command_option(forwarded_args, flag, ''):
                forwarded_args += [flag, str(resource_plan['stage_cpus'])]
    else:
        args.jobs = args.jobs or 1
    subject_commands = [
        (
            subject,
            build_subject_command(
                subject,
                pipeline_script=args.pipeline_script,
                python_executable=args.python_executable,
                output_root=output_base if args.output_root else None,
                run_name=args.run_name,
                execute=bool(args.execute),
                forwarded_args=[*forwarded_args, *(resource_plan or {}).get('subject_arguments', {}).get(subject.subject_id, [])],
            ),
        )
        for subject in runnable_subjects
    ]

    planned_commands = [
        build_subject_command(
            subject,
            pipeline_script=args.pipeline_script,
            python_executable=args.python_executable,
            output_root=output_base if args.output_root else None,
            run_name=args.run_name,
            execute=bool(args.execute),
            forwarded_args=[*forwarded_args, *(resource_plan or {}).get('subject_arguments', {}).get(subject.subject_id, [])],
        )
        for subject in runnable_subjects
    ]

    result_payloads: list[dict[str, object]] = []
    schedule_dir, child_environment = None, None
    if resource_mode:
        from shared.cohort_schedule import create_schedule, ENV
        from uuid import uuid4
        try:
            schedule_dir = create_schedule(manifest_path.parent / ('schedule_' + uuid4().hex),
                **{key: resource_plan[key] for key in ('cpu_budget', 'memory_gb', 'stage_cpus',
                       'stage_memory_gb', 'gpu_slots', 'subject_profiles')}, planning=resource_plan,
                policy=resource_plan['policy'], io_slots=resource_plan['io_slots'],
                history_path=resource_plan['history_path'] if args.scheduler == 'adaptive' and args.execute else None)
        except (ValueError, ImportError) as exc:
            parser.error(str(exc))
        child_environment = {ENV: str(schedule_dir), 'SLICERDSPACE_STAGE_CPUS': str(resource_plan['stage_cpus'])}
        child_environment.update({key: str(resource_plan['stage_cpus']) for key in (
            'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', 'MRTRIX_NTHREADS')})
    failures = 0
    started = time.time()
    if subject_commands:
        pending_subjects = iter(subject_commands)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            future_to_subject: dict[concurrent.futures.Future[SubjectRunResult], BatchSubject] = {}

            def submit_next() -> bool:
                try:
                    subject, command = next(pending_subjects)
                except StopIteration:
                    return False
                kwargs = {'environment': child_environment} if child_environment is not None else {}
                future_to_subject[executor.submit(_run_subject, command, subject, log_dir, **kwargs)] = subject
                return True

            for _ in range(min(args.jobs, len(subject_commands))):
                submit_next()

            stop_submitting = False
            while future_to_subject:
                done, _pending = concurrent.futures.wait(
                    future_to_subject,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    future_to_subject.pop(future)
                    result = future.result()
                    unsuccessful = result.run_status in {'failed', 'cancelled', 'unknown'} or (
                        result.run_status == 'completed_with_blocks' and not args.allow_partial_success)
                    if result.returncode != 0 or unsuccessful:
                        failures += 1
                        stop_submitting = not args.continue_on_error
                    result_payloads.append(
                        {
                            "subject_id": result.subject_id,
                            "subject_root": result.subject_root.as_posix(),
                            "returncode": result.returncode,
                            "elapsed_seconds": result.elapsed_seconds,
                            "command": result.command,
                            "stdout_log": result.stdout_log.as_posix(),
                            "stderr_log": result.stderr_log.as_posix(),
                            "status": result.run_status,
                            "status_source": result.status_source,
                            "run_manifest_path": result.run_manifest_path.as_posix() if result.run_manifest_path else None,
                            "task_counts": result.task_counts,
                            "manifest_error": result.manifest_error,
                        }
                    )
                # Drain active processes after a failure so their outputs and
                # exit states remain in the batch record; stop only new launches.
                while not stop_submitting and len(future_to_subject) < args.jobs and submit_next():
                    pass

    subject_counts: dict[str, int] = {}
    task_counts: dict[str, int] = {}
    for item in result_payloads:
        state = str(item['status'])
        subject_counts[state] = subject_counts.get(state, 0) + 1
        for state, count in (item['task_counts'] or {}).items():
            task_counts[state] = task_counts.get(state, 0) + count
    batch_status = ('empty' if not result_payloads else 'failed' if subject_counts.get('failed') else
                    'cancelled' if subject_counts.get('cancelled') else 'incomplete' if subject_counts.get('unknown') else
                    'completed_with_blocks' if subject_counts.get('completed_with_blocks') else
                    'completed' if args.execute else 'planned')
    from shared.cohort_schedule import schedule_summary
    payload: dict[str, object] = {
        "kind": "slicerdsurfer_batch_run",
        "version": 2,
        "batch_root": args.batch_root.expanduser().resolve().as_posix(),
        "output_root": output_base.as_posix() if args.output_root else None,
        "run_name": args.run_name,
        "execute": bool(args.execute),
        "jobs": args.jobs,
        "scheduler": args.scheduler,
        "subject_limit": args.subject_limit,
        "resource_schedule": str(schedule_dir) if schedule_dir else None,
        "resource_plan": resource_plan,
        "resource_measurements": schedule_summary(schedule_dir) if schedule_dir else None,
        "forwarded_args": forwarded_args,
        "arrange_inputs": args.arrange_inputs,
        "subjects": [
            {
                "subject_id": subject.subject_id,
                "subject_root": subject.subject_root.as_posix(),
                "source": subject.source,
                "mode": subject.mode,
            }
            for subject in subjects
        ],
        "skipped": skipped,
        "planned_commands": planned_commands,
        "results": sorted(result_payloads, key=lambda item: str(item["subject_id"])),
        "not_launched": [subject.subject_id for subject in runnable_subjects
                         if subject.subject_id not in {str(item["subject_id"]) for item in result_payloads}],
        "status": batch_status,
        "subject_status_counts": subject_counts,
        "task_status_counts": task_counts,
        "allow_partial_success": bool(args.allow_partial_success),
        "elapsed_seconds": round(time.time() - started, 3),
        "manifest_path": "",
    }
    written_manifest = _write_manifest(manifest_path, payload)
    payload["manifest_path"] = written_manifest.as_posix()
    _write_manifest(written_manifest, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    # Machine-readable line for callers (e.g. the Slicer widget's background
    # QProcess reader) that watch stdout for a manifest path: unlike the
    # single-subject pipeline CLI's manifest, this file is not named
    # "*_orchestration_manifest.json", so it needs an explicit marker.
    print(f"[Batch] manifest: {written_manifest.as_posix()}")
    print(written_manifest.as_posix(), file=sys.stderr)

    if failures:
        return 2
    if not subjects:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
