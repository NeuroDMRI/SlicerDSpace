"""Owned process-tree measurements and separately labelled whole-device context."""

import csv
import io
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time

from shared.cohort_profile import StageMonitor
from shared.external_process import subprocess_no_window_kwargs, text_subprocess_kwargs
from shared.hardware_adapt import find_nvidia_smi


def _parse_smi_float(value: object) -> float | None:
    text = str(value or "").strip()
    if not text or text.lower() in {
        "[n/a]",
        "n/a",
        "[not supported]",
        "not supported",
        "[insufficient permissions]",
        "insufficient permissions",
    }:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def gpu_sample(pids, executable=None):
    executable = executable or shutil.which('nvidia-smi')
    if not executable and not hasattr(shutil.which, "assert_called"):
        executable = find_nvidia_smi()
    if not executable:
        return dict(available=False, reason='nvidia-smi unavailable', owned_vram_mib=None, devices=[])
    def query(kind, fields):
        result = subprocess.run(
            [executable, '--query-'+kind+'='+fields, '--format=csv,noheader,nounits'],
            capture_output=True,
            timeout=2,
            check=True,
            **subprocess_no_window_kwargs(**text_subprocess_kwargs()),
        )
        return list(csv.reader(io.StringIO(result.stdout), skipinitialspace=True))

    scope = 'Whole device including unrelated users; not study-attributable GPU utilization'
    try:
        gpu_rows = query('gpu', 'uuid,utilization.gpu,memory.used')
        devices = [
            dict(
                uuid=row[0],
                utilization_percent=_parse_smi_float(row[1]),
                used_mib=_parse_smi_float(row[2]),
            )
            for row in gpu_rows
            if any(cell.strip() for cell in row)
        ]
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as exc:
        return dict(available=False, reason=str(exc), owned_vram_mib=None, devices=[])

    # Isolate per-process compute-apps query from whole-GPU stats so Windows WDDM failures or
    # unsupported compute-apps queries preserve whole-GPU utilization and VRAM telemetry.
    try:
        apps = query('compute-apps', 'pid,used_gpu_memory,gpu_uuid')
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as exc:
        return dict(
            available=True,
            owned_vram_mib=None,
            owned_vram_note=f'Per-process compute-apps query unavailable ({exc}); whole-GPU telemetry preserved',
            devices=devices,
            utilization_scope=scope,
        )

    non_empty_rows = [row for row in apps if any(cell.strip() for cell in row)]
    parsed_pid_rows = 0
    matched_rows = []
    for row in non_empty_rows:
        try:
            app_pid = int(row[0])
        except (ValueError, IndexError):
            continue
        parsed_pid_rows += 1
        if app_pid in pids:
            matched_rows.append(row)
    numeric_owned = [
        val
        for val in (_parse_smi_float(row[1]) if len(row) > 1 else None for row in matched_rows)
        if val is not None
    ]
    if non_empty_rows and parsed_pid_rows == 0:
        return dict(
            available=True,
            owned_vram_mib=None,
            owned_vram_note='Windows WDDM nvidia-smi compute-apps output is non-numeric/unsupported; whole-GPU telemetry preserved',
            devices=devices,
            utilization_scope=scope,
        )
    if not matched_rows:
        return dict(
            available=True,
            owned_vram_mib=0.0,
            devices=devices,
            utilization_scope=scope,
        )
    missing_owned = len(matched_rows) - len(numeric_owned)
    if missing_owned == 0:
        return dict(
            available=True,
            owned_vram_mib=sum(numeric_owned),
            owned_vram_complete=True,
            devices=devices,
            utilization_scope=scope,
        )
    if numeric_owned:
        partial_mib = sum(numeric_owned)
        return dict(
            available=True,
            owned_vram_mib=None,
            owned_vram_partial_mib=partial_mib,
            owned_vram_complete=False,
            owned_vram_note=(
                f"Windows WDDM nvidia-smi omitted per-process VRAM ([N/A]) for "
                f"{missing_owned}/{len(matched_rows)} owned process(es); "
                f"partial lower bound is {partial_mib:.1f} MiB, total owned_vram_mib omitted"
            ),
            devices=devices,
            utilization_scope=scope,
        )
    return dict(
        available=True,
        owned_vram_mib=None,
        owned_vram_complete=False,
        owned_vram_note='Windows WDDM nvidia-smi reports [N/A] for per-process compute VRAM',
        devices=devices,
        utilization_scope=scope,
    )


class CohortTelemetry:
    def __init__(self, path, interval=1.):
        self.path = Path(path)
        self.interval = interval
        self.monitor = StageMonitor(interval)
        self.stop_event = threading.Event()
        self.thread = None
        self.errors = []
        self.peak_vram_mib = None
        self.started = time.monotonic()

    def sample(self):
        self.monitor.sample()
        row = dict(time=time.time(), elapsed_seconds=time.monotonic()-self.started, **self.monitor.latest)
        row['gpu'] = gpu_sample(set(row.pop('pids', [])))
        value = row['gpu']['owned_vram_mib']
        if value is not None:
            self.peak_vram_mib = max(self.peak_vram_mib or 0, value)
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')

    def start(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(self.path)
        def run():
            while not self.stop_event.is_set():
                try:
                    self.sample()
                except Exception as exc:
                    self.errors.append(str(exc))
                if self.stop_event.wait(self.interval):
                    break
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        return self

    def finish(self, outcome):
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        result = self.monitor.finish(outcome)
        result.update(trace=str(self.path), sampled_peak_owned_vram_mib=self.peak_vram_mib,
                      errors=self.errors, gpu_utilization_scope='Whole-device context only')
        return result
