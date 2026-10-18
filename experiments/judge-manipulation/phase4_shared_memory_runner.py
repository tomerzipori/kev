"""Run the amended Phase 4 GPU probe under a Windows and WSL resource watchdog."""

import argparse
import hashlib
import json
import math
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from phase4_shared_memory_support import atomic_json


PLAN_SHA256 = "0329f3bc4fc6e9779caf1b32b393729b9f36fc6d28de85bb17e7c1b3bae4b48a"
MODEL_LOCK_SHA256 = "0bdce0513f5d85c09154f6a8808ad973bc2848d679d1144c7093e64dc6d8fb19"
FIXTURE_SHA256 = "ccc4c494c47e2559334b93842ce30c218da0e0fb6234eaf8d111aa2c5c97be20"
SOURCE_FILES = (
    "experiments/judge-manipulation/phase4_shared_memory_prepare.py",
    "experiments/judge-manipulation/phase4_shared_memory_runner.py",
    "experiments/judge-manipulation/phase4_shared_memory_windows_sample.ps1",
    "experiments/judge-manipulation/phase4_shared_memory_gate.py",
    "experiments/judge-manipulation/phase4_shared_memory_kernels.py",
    "experiments/judge-manipulation/phase4_shared_memory_reload.py",
    "experiments/judge-manipulation/phase4_shared_memory_support.py",
    "experiments/judge-manipulation/phase4_shared_memory_train.py",
    "experiments/judge-manipulation/phase4_model_evidence.py",
    "experiments/judge-manipulation/phase4-shared-memory-plan.json",
    "experiments/judge-manipulation/phase4-shared-memory-overlay-lock.json",
    "experiments/judge-manipulation/phase4-shared-memory-overlay-supplement.json",
    "experiments/judge-manipulation/model-lock-phase-4.json",
)
HISTORICAL_FILES = (
    "experiments/judge-manipulation/model-lock-phase-4.json",
    "experiments/judge-manipulation/report-phase-4.json",
    "experiments/judge-manipulation/report-phase-4.md",
    "experiments/judge-manipulation/feasibility-gate-phase-4.json",
    "experiments/judge-manipulation/feasibility-gate-phase-4.md",
    "experiments/judge-manipulation/memory-gate-stop-phase-4.json",
    "experiments/judge-manipulation/HANDOFF-phase-4.md",
    "experiments/judge-manipulation/AMENDMENT-shared-memory-phase-4.json",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def process_info(pid):
    stat_path = Path(f"/proc/{pid}/stat")
    raw = stat_path.read_text(encoding="utf-8")
    end = raw.rfind(")")
    if end < 0:
        raise RuntimeError(f"cannot parse Linux process identity for PID {pid}")
    fields = raw[end + 2:].split()
    status = fields[0]
    parent_pid, process_group, session_id, start_time = int(fields[1]), int(fields[2]), int(fields[3]), int(fields[19])
    command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
    return {"pid": pid, "ppid": parent_pid, "pgrp": process_group, "session_id": session_id,
            "starttime_ticks": start_time,
            "state": status, "command": command}


def process_group_members(process_group):
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            info = process_info(int(entry.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, OSError):
            continue
        if info["pgrp"] == process_group:
            status = {}
            try:
                for line in (entry / "status").read_text(encoding="utf-8").splitlines():
                    if line.startswith(("VmRSS:", "VmHWM:")):
                        key, value = line.split(":", 1)
                        status[key] = int(value.strip().split()[0]) * 1024
            except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
                pass
            members.append({**info, "rss_bytes": status.get("VmRSS"), "peak_rss_bytes": status.get("VmHWM")})
    return members


def memory_info():
    values = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.strip().split()[0]) * 1024
    required = ("MemAvailable", "SwapTotal", "SwapFree")
    if any(name not in values for name in required):
        raise RuntimeError("WSL memory counters are incomplete")
    return {"wsl_memavailable_bytes": values["MemAvailable"],
            "wsl_swap_used_bytes": max(0, values["SwapTotal"] - values["SwapFree"]),
            "wsl_swap_total_bytes": values["SwapTotal"]}


def gpu_memory_info(timeout_seconds):
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        timeout=max(0.001, timeout_seconds),
    )
    if result.returncode:
        raise RuntimeError(f"nvidia-smi memory query failed: {result.stderr.strip()}")
    rows = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            raise RuntimeError(f"unrecognized nvidia-smi memory row: {line!r}")
        name, total, used, free = parts
        if not name or any(not value.isdecimal() for value in (total, used, free)):
            raise RuntimeError(f"unrecognized nvidia-smi memory row: {line!r}")
        rows.append({"name": name, "total_bytes": int(total) * 1024**2,
                     "used_bytes": int(used) * 1024**2, "free_bytes": int(free) * 1024**2})
    if len(rows) != 1 or "RTX 4060 Ti" not in rows[0]["name"]:
        raise RuntimeError(f"expected one registered RTX 4060 Ti, found {rows}")
    return {**rows[0], "device_count": len(rows)}


def child_observation(paths):
    for path in paths:
        try:
            return read_json(path), str(path)
        except (OSError, json.JSONDecodeError):
            continue
    return None, None


class WindowsSampler:
    def __init__(self, script, runtime_dir, interval_seconds):
        self.script = Path(script)
        self.runtime_dir = Path(runtime_dir)
        self.interval_seconds = interval_seconds
        self.stop_path = self.runtime_dir / "windows-sampler.stop"
        self.items = queue.Queue()
        self.process = None
        self.reader = None
        self.stderr_stream = None
        self.reader_done = False

    @staticmethod
    def _windows_path(path):
        result = subprocess.run(["wslpath", "-w", str(path)], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if result.returncode:
            raise RuntimeError(f"wslpath could not convert {path}: {result.stderr.strip()}")
        return result.stdout.strip()

    @staticmethod
    def _powershell():
        candidate = shutil.which("powershell.exe")
        if candidate:
            return candidate
        fallback = Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
        if fallback.is_file():
            return str(fallback)
        raise RuntimeError("Windows PowerShell is unavailable from this WSL session")

    def start(self, timeout_seconds):
        distro = os.environ.get("WSL_DISTRO_NAME", "Ubuntu-24.04")
        stop_unc = f"\\\\wsl.localhost\\{distro}\\{self.stop_path.as_posix().lstrip('/').replace('/', chr(92))}"
        self.stderr_stream = (self.runtime_dir / "windows-sampler.stderr.log").open("x", encoding="utf-8")
        self.process = subprocess.Popen(
            [self._powershell(), "-NoLogo", "-NoProfile", "-NonInteractive", "-File", self._windows_path(self.script),
             "-StopPath", stop_unc, "-IntervalMilliseconds", str(round(self.interval_seconds * 1000))],
            stdout=subprocess.PIPE, stderr=self.stderr_stream, text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        self.reader = threading.Thread(target=self._read, name="phase4-windows-counter-reader", daemon=True)
        self.reader.start()
        try:
            return self.items.get(timeout=max(0.1, timeout_seconds))
        except queue.Empty as error:
            raise RuntimeError("Windows resource sampler did not return its first sample before the deadline") from error

    def _read(self):
        try:
            for line in self.process.stdout:
                self.items.put((time.monotonic_ns(), line.rstrip("\r\n")))
        finally:
            self.reader_done = True
            self.items.put((time.monotonic_ns(), None))

    def get(self, timeout):
        try:
            return self.items.get(timeout=max(0, timeout))
        except queue.Empty:
            return None

    def stop(self):
        if self.process is None:
            return {"stopped": True, "exit_code": None}
        self.stop_path.write_text("stop\n", encoding="utf-8")
        try:
            code = self.process.wait(timeout=6)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                code = self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                code = self.process.wait(timeout=2)
        if self.reader:
            self.reader.join(timeout=2)
        if self.stderr_stream:
            self.stderr_stream.close()
        return {"stopped": self.process.poll() is not None, "exit_code": code,
                "reader_stopped": self.reader is None or not self.reader.is_alive(),
                "pid": self.process.pid}


class ResourceMonitor:
    def __init__(self, plan, runtime_dir):
        self.limits = plan["limits"]
        self.runtime_dir = Path(runtime_dir)
        self.sample_path = self.runtime_dir / "monitoring-samples.jsonl"
        self.stream = self.sample_path.open("x", encoding="utf-8", newline="\n")
        self.samples = []
        self.resource_times = []
        self.next_missing_ns = None
        self.max_missing_streak = self.max_violation_streak = 0
        self.missing_streak = self.violation_streak = 0
        self.missing_count = self.violation_count = 0
        self.stop_reason = None
        self.active_process_group = None
        self.observation_paths = []
        self.current_stage_path = None
        self.latest_stage = None
        self.latest_observation = None

    def close(self):
        self.stream.flush()
        self.stream.close()

    def _source_state(self):
        stage = None
        observation = None
        try:
            stage = read_json(self.current_stage_path) if self.current_stage_path else None
        except (OSError, json.JSONDecodeError):
            pass
        observation, observation_path = child_observation(self.observation_paths)
        cuda_active = bool((stage or {}).get("cuda_active") or (observation or {}).get("cuda_active"))
        self.latest_stage, self.latest_observation = stage, observation
        return stage, observation, observation_path, cuda_active

    def _append(self, sample):
        timestamp = sample["monotonic_ns"]
        if sample.get("sample_kind", "resource_poll") == "resource_poll":
            self.resource_times.append(timestamp)
        missing = False
        required = ("windows_adapter_shared_bytes", "active_model_process_rss_bytes",
                    "windows_available_physical_bytes", "wsl_memavailable_bytes", "wsl_swap_used_bytes",
                    "gpu_dedicated_used_bytes", "gpu_dedicated_free_bytes")
        for name in required:
            value = sample.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                missing = True
        if not isinstance(sample.get("cuda_active"), bool):
            missing = True
        if sample.get("cuda_active"):
            for name in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
                value = sample.get(name)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    missing = True
        violation = False
        for name, direction, key in (
            ("cuda_peak_allocated_bytes", "max", "cuda_peak_allocated_bytes_max"),
            ("cuda_peak_reserved_bytes", "max", "cuda_peak_reserved_bytes_max"),
            ("windows_adapter_shared_bytes", "max", "windows_adapter_shared_bytes_max"),
            ("active_model_process_rss_bytes", "max", "active_model_process_rss_bytes_max"),
            ("windows_available_physical_bytes", "min", "windows_available_physical_bytes_min"),
            ("wsl_memavailable_bytes", "min", "wsl_memavailable_bytes_min"),
            ("wsl_swap_used_bytes", "max", "wsl_swap_used_bytes_max"),
        ):
            if name.startswith("cuda_") and sample.get("cuda_active") is not True:
                continue
            value = sample.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                continue
            limit = self.limits[key]
            violation = violation or (value > limit if direction == "max" else value < limit)
        self.missing_streak = self.missing_streak + 1 if missing else 0
        self.violation_streak = self.violation_streak + 1 if violation else 0
        self.max_missing_streak = max(self.max_missing_streak, self.missing_streak)
        self.max_violation_streak = max(self.max_violation_streak, self.violation_streak)
        self.missing_count += int(missing)
        self.violation_count += int(violation)
        sample["critical_monitoring_missing"] = missing
        sample["resource_limit_violation"] = violation
        self.samples.append(sample)
        self.stream.write(json.dumps(sample, sort_keys=True, separators=(",", ":")) + "\n")
        self.stream.flush()
        threshold = self.limits["consecutive_violating_samples"]
        if self.missing_streak >= threshold:
            self.stop_reason = self.stop_reason or "monitoring_failure"
        if self.violation_streak >= threshold:
            self.stop_reason = self.stop_reason or "resource_limit"

    def add_windows_sample(self, item, *, process_group=None, gpu_query_timeout_seconds=0.5):
        timestamp, line = item
        try:
            windows = json.loads(line) if line is not None else {"status": "sampler_stopped"}
        except json.JSONDecodeError as error:
            windows = {"status": "invalid_json", "error": str(error), "raw": line}
        linux, gpu, group_members = {}, {}, []
        try:
            linux = memory_info()
        except (OSError, ValueError, RuntimeError) as error:
            linux = {"wsl_memavailable_bytes": None, "wsl_swap_used_bytes": None,
                     "memory_error": f"{type(error).__name__}: {error}"}
        try:
            gpu = gpu_memory_info(gpu_query_timeout_seconds)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            gpu = {"used_bytes": None, "free_bytes": None,
                   "gpu_query_error": f"{type(error).__name__}: {error}"}
        if process_group:
            group_members = process_group_members(process_group)
        rss = sum(member["rss_bytes"] for member in group_members if isinstance(member.get("rss_bytes"), int))
        peak_rss = max((member["peak_rss_bytes"] for member in group_members
                        if isinstance(member.get("peak_rss_bytes"), int)), default=0)
        stage, observation, observation_path, cuda_active = self._source_state()
        sample = {
            "sample_kind": "resource_poll", "monotonic_ns": timestamp,
            "observed_utc": windows.get("observed_utc"), "watchdog_received_utc": datetime.now(timezone.utc).isoformat(),
            "stage": (stage or {}).get("stage"), "stage_cuda_active": (stage or {}).get("cuda_active"),
            "cuda_active": cuda_active, "active_model_process_rss_bytes": rss,
            "active_model_process_peak_rss_bytes": peak_rss,
            "active_model_process_pids": [member["pid"] for member in group_members],
            "windows_sampler_status": windows.get("status"),
            "windows_adapter_shared_bytes": windows.get("windows_adapter_shared_bytes"),
            "windows_adapter_dedicated_bytes": windows.get("windows_adapter_dedicated_bytes"),
            "windows_available_physical_bytes": windows.get("windows_available_physical_bytes"),
            "windows_adapter_counters": windows.get("windows_adapter_counters", []),
            "windows_gpu_process_counters": windows.get("windows_gpu_process_counters", []),
            **linux,
            "gpu_name": gpu.get("name"), "gpu_total_bytes": gpu.get("total_bytes"),
            "gpu_dedicated_used_bytes": gpu.get("used_bytes"), "gpu_dedicated_free_bytes": gpu.get("free_bytes"),
            "gpu_query_error": gpu.get("gpu_query_error"),
            "cuda_observation_path": observation_path,
            "cuda_current_allocated_bytes": (observation or {}).get("cuda_current_allocated_bytes"),
            "cuda_peak_allocated_bytes": (observation or {}).get("cuda_peak_allocated_bytes"),
            "cuda_peak_reserved_bytes": (observation or {}).get("cuda_peak_reserved_bytes"),
            "cuda_observation_monotonic_ns": (observation or {}).get("monotonic_ns"),
            "cuda_allocator_status": (observation or {}).get("allocator_status"),
            "windows_error": windows.get("error"),
        }
        self._append(sample)
        self.next_missing_ns = timestamp + int(self.limits["poll_interval_seconds"] * 1e9)

    def add_missing_opportunity(self, timestamp):
        stage, observation, observation_path, cuda_active = self._source_state()
        sample = {
            "sample_kind": "missed_poll", "monotonic_ns": timestamp,
            "observed_utc": datetime.now(timezone.utc).isoformat(), "cuda_active": cuda_active,
            "stage": (stage or {}).get("stage"), "stage_cuda_active": (stage or {}).get("cuda_active"),
            "active_model_process_rss_bytes": None,
            "windows_adapter_shared_bytes": None, "windows_available_physical_bytes": None,
            "wsl_memavailable_bytes": None, "wsl_swap_used_bytes": None,
            "gpu_dedicated_used_bytes": None, "gpu_dedicated_free_bytes": None,
            "cuda_peak_allocated_bytes": (observation or {}).get("cuda_peak_allocated_bytes"),
            "cuda_peak_reserved_bytes": (observation or {}).get("cuda_peak_reserved_bytes"),
            "cuda_observation_path": observation_path,
            "monitoring_error": "Windows resource sampler produced no sample at the scheduled poll time",
        }
        self._append(sample)

    def add_due_missing_samples(self, now_ns):
        if self.next_missing_ns is None:
            return
        interval = int(self.limits["poll_interval_seconds"] * 1e9)
        while now_ns >= self.next_missing_ns:
            due = self.next_missing_ns
            self.add_missing_opportunity(due)
            self.next_missing_ns = due + interval

    def summary(self):
        gaps = [(right - left) / 1e9 for left, right in zip(self.resource_times, self.resource_times[1:])]
        return {
            "status": "complete" if self.samples and self.stop_reason is None else "failed",
            "poll_interval_seconds": self.limits["poll_interval_seconds"],
            "samples": self.samples,
            "resource_poll_sample_count": len(self.resource_times),
            "max_gap_seconds": max(gaps, default=None),
            "max_consecutive_missing_samples": self.max_missing_streak,
            "missing_sample_count": self.missing_count,
            "max_consecutive_violating_samples": self.max_violation_streak,
            "violating_sample_count": self.violation_count,
        }


class ChildProcess:
    def __init__(self, process, command, script_path):
        self.process = process
        self.command = list(command)
        self.pid = process.pid
        self.pgid = process.pid
        self.script_path = str(Path(script_path).resolve())
        deadline = time.monotonic() + 2
        while True:
            info = process_info(self.pid)
            if info["pgrp"] == self.pgid and self.script_path in info["command"]:
                self.identity = {"pid": self.pid, "process_group": self.pgid,
                                 "session_id": info["session_id"],
                                 "starttime_ticks": info["starttime_ticks"], "initial_command": info["command"]}
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f"benchmark child identity did not match its isolated process group: {info}")
            time.sleep(0.02)

    def signal(self, value):
        try:
            info = process_info(self.pid)
        except FileNotFoundError:
            members = process_group_members(self.pgid)
            mismatched = [member for member in members
                          if member["session_id"] != self.identity["session_id"]]
            if mismatched:
                raise RuntimeError("process-group id is occupied by members outside the verified child session")
            if not members:
                return {"signal": value, "sent": False, "reason": "owned_process_group_empty"}
            os.killpg(self.pgid, value)
            return {"signal": value, "sent": True, "leader_exited": True,
                    "owned_members": [member["pid"] for member in members]}
        if (info["pgrp"] != self.pgid or info["session_id"] != self.identity["session_id"]
                or info["starttime_ticks"] != self.identity["starttime_ticks"]):
            raise RuntimeError("refusing to signal a process group whose verified child identity changed")
        os.killpg(self.pgid, value)
        return {"signal": value, "sent": True, "leader": info}

    def empty(self):
        members = process_group_members(self.pgid)
        if any(member["session_id"] != self.identity["session_id"] for member in members):
            raise RuntimeError("process-group id is occupied by members outside the verified child session")
        return not members


class Runner:
    def __init__(self, repo, run_dir):
        self.repo = Path(repo).resolve()
        self.run_dir = Path(run_dir).resolve()
        self.experiments = self.repo / "experiments/judge-manipulation"
        self.setup_path = self.run_dir / "setup.json"
        self.setup = read_json(self.setup_path)
        if Path(self.setup["runtime_directory"]).resolve() != self.run_dir:
            raise RuntimeError("setup receipt belongs to a different Linux runtime directory")
        if self.setup.get("status") != "cpu_preflight_passed":
            raise RuntimeError("CPU preflight has not passed")
        self.plan_path = self.experiments / "phase4-shared-memory-plan.json"
        self.plan = read_json(self.plan_path)
        if sha256(self.plan_path) != PLAN_SHA256:
            raise RuntimeError("approved plan bytes changed before the benchmark")
        if sha256(self.experiments / "model-lock-phase-4.json") != MODEL_LOCK_SHA256:
            raise RuntimeError("frozen model lock bytes changed before the benchmark")
        current_sources = {name: sha256(self.repo / name) for name in SOURCE_FILES}
        if current_sources != self.setup.get("source_hashes_before"):
            raise RuntimeError("benchmark sources changed after CPU preflight")
        if self.setup["fixture"].get("fixture_sha256") != FIXTURE_SHA256:
            raise RuntimeError("CPU preflight fixture hash does not match the registered fixture")
        if not Path(self.setup["fixture"]["fixture_path"]).is_file():
            raise RuntimeError("CPU preflight fixture file is missing")
        self.run_id = self.setup["run_id"]
        self.overlay = Path(self.setup["overlay"]["path"])
        self.started = time.monotonic()
        self.started_ns = time.monotonic_ns()
        self.started_utc = datetime.now(timezone.utc).isoformat()
        self.deadline = self.started + self.plan["limits"]["wall_seconds"]
        self.stop_reason = None
        self.active_child = None
        self.child_groups = []
        self.stage_results = {}
        self.stage_outputs = {}
        self.monitor = ResourceMonitor(self.plan, self.run_dir)
        self.sampler = WindowsSampler(self.experiments / "phase4_shared_memory_windows_sample.ps1",
                                      self.run_dir, self.plan["limits"]["poll_interval_seconds"])
        self.repo_retained_run = self.repo / "runs/judge-manipulation-phase4-shared" / self.run_id
        self.repo_retained_report = self.experiments / "shared-memory-reports" / self.run_id
        if self.repo_retained_run.exists() or self.repo_retained_report.exists():
            raise RuntimeError("retained run or report path already exists")
        if (self.run_dir / "report.json").exists() or (self.run_dir / "gate-verdict.json").exists():
            raise RuntimeError("runtime report path already exists")

    def _record_stage(self, name, details):
        details = {"stage": name, "updated_utc": datetime.now(timezone.utc).isoformat(),
                   "monotonic_ns": time.monotonic_ns(), **details}
        atomic_json(self.run_dir / "runner-stage.json", details)

    def _environment(self):
        env = os.environ.copy()
        prior = env.get("PYTHONPATH", "")
        paths = [str(self.overlay), str(self.repo), str(self.experiments)]
        if prior:
            paths.append(prior)
        env["PYTHONPATH"] = os.pathsep.join(paths)
        env["PYTHONUNBUFFERED"] = "1"
        return env

    def _poll_monitor(self, timeout=0.1):
        item = self.sampler.get(timeout)
        if item is not None:
            self.monitor.add_windows_sample(item,
                                            process_group=self.active_child.pgid if self.active_child else None,
                                            gpu_query_timeout_seconds=min(
                                                self.plan["limits"]["poll_interval_seconds"] * 0.5,
                                                max(0.001, self.deadline - time.monotonic())))
        self.monitor.add_due_missing_samples(time.monotonic_ns())
        if self.monitor.stop_reason:
            self.stop_reason = self.stop_reason or self.monitor.stop_reason
        return item

    def _wait_child(self, child, grace_seconds):
        until = min(self.deadline, time.monotonic() + grace_seconds)
        while time.monotonic() < until:
            if child.process.poll() is not None:
                return True
            if time.monotonic() >= self.deadline:
                return False
            self._poll_monitor(0.05)
        return child.process.poll() is not None

    def _stop_child(self, child, *, hard=False):
        signals = []
        if child.process.poll() is not None and child.empty():
            child.process.wait(timeout=0)
            return {"signals": signals, "return_code": child.process.returncode, "process_group_empty": True}
        if hard or time.monotonic() >= self.deadline:
            signals.append(child.signal(signal.SIGKILL))
        else:
            signals.append(child.signal(signal.SIGINT))
            if not self._wait_child(child, 30):
                if time.monotonic() >= self.deadline:
                    signals.append(child.signal(signal.SIGKILL))
                else:
                    signals.append(child.signal(signal.SIGTERM))
                    if not self._wait_child(child, 30):
                        signals.append(child.signal(signal.SIGKILL))
        try:
            child.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            signals.append(child.signal(signal.SIGKILL))
            child.process.wait(timeout=5)
        empty_deadline = min(self.deadline, time.monotonic() + 5)
        while not child.empty() and time.monotonic() < empty_deadline:
            self._poll_monitor(0.05)
        if not child.empty():
            signals.append(child.signal(signal.SIGKILL))
            try:
                child.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                return {"signals": signals, "return_code": child.process.poll(), "process_group_empty": False,
                        "error": "child leader did not exit after SIGKILL within five seconds"}
        return {"signals": signals, "return_code": child.process.returncode, "process_group_empty": child.empty()}

    def run_stage(self, name, script, command, output_paths, observation_path, stage_path):
        if self.stop_reason:
            return {"status": "not_started", "reason": self.stop_reason}
        if time.monotonic() >= self.deadline:
            self.stop_reason = "wall_deadline"
            return {"status": "failed", "reason": self.stop_reason}
        self.monitor.current_stage_path = stage_path
        self.monitor.observation_paths = [observation_path]
        log_path = self.run_dir / f"{name}.log"
        command_path = self.run_dir / f"{name}-command.json"
        if any(Path(path).exists() for path in output_paths) or log_path.exists() or command_path.exists():
            raise RuntimeError(f"refusing to overwrite {name} stage evidence")
        command_path.write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
        started = time.monotonic()
        stage_started_utc = datetime.now(timezone.utc).isoformat()
        self._record_stage(name, {"status": "running", "script": str(script), "command": command,
                                  "started_utc": stage_started_utc})
        with log_path.open("x", encoding="utf-8", newline="\n") as log:
            process = subprocess.Popen(command, cwd=self.repo, env=self._environment(), stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True, text=True)
            try:
                child = ChildProcess(process, command, script)
            except BaseException:
                initial_identity = None
                try:
                    info = process_info(process.pid)
                    if info["pgrp"] == process.pid and str(Path(script).resolve()) in info["command"]:
                        initial_identity = {"pid": process.pid, "process_group": process.pid,
                                            "session_id": info["session_id"],
                                            "starttime_ticks": info["starttime_ticks"],
                                            "initial_command": info["command"]}
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                except FileNotFoundError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                members = process_group_members(process.pid)
                self.child_groups.append({
                    "stage": name, "identity": initial_identity, "log": str(log_path),
                    "cleanup": {"return_code": process.poll(), "process_group_empty": not members,
                                "identity_verified": initial_identity is not None},
                })
                raise
            self.active_child = child
            group_record = {"stage": name, "identity": child.identity, "log": str(log_path), "cleanup": None}
            self.child_groups.append(group_record)
            stage_stop = None
            while True:
                now = time.monotonic()
                return_code = process.poll()
                if return_code is not None:
                    break
                if now >= self.deadline:
                    stage_stop = "wall_deadline"
                    self.stop_reason = self.stop_reason or stage_stop
                    break
                if self.monitor.stop_reason:
                    stage_stop = self.monitor.stop_reason
                    self.stop_reason = self.stop_reason or stage_stop
                    break
                self._poll_monitor(0.05)
            cleanup = None
            if process.poll() is None:
                cleanup = self._stop_child(child, hard=stage_stop == "wall_deadline")
            else:
                process.wait()
                cleanup = {"signals": [], "return_code": process.returncode, "process_group_empty": child.empty()}
                if not cleanup["process_group_empty"]:
                    cleanup.update(self._stop_child(child, hard=time.monotonic() >= self.deadline))
            group_record["cleanup"] = cleanup
            self.active_child = None
        elapsed = round(time.monotonic() - started, 6)
        result_path = next((Path(path) for path in output_paths if Path(path).is_file()), None)
        stage_payload = read_json(result_path) if result_path else None
        code = process.returncode
        status = "passed" if code == 0 and stage_payload and stage_payload.get("status") == "passed" and not stage_stop and cleanup["process_group_empty"] else "failed"
        if status != "passed" and self.stop_reason is None:
            if name == "kernel_correctness":
                self.stop_reason = "kernel_correctness_failure"
            else:
                self.stop_reason = f"{name}_failure"
        result = {
            "status": status, "fresh_process": name in {"training", "reload"},
            "separate_process": name == "kernel_correctness", "started_utc": stage_started_utc,
            "elapsed_seconds": elapsed, "script": str(script), "command": command,
            "return_code": code, "log": str(log_path), "result_path": str(result_path) if result_path else None,
            "result": stage_payload, "process_group": child.identity, "cleanup": cleanup,
        }
        self.stage_outputs[name] = stage_payload
        self.stage_results[name] = result
        self._record_stage(name, {"status": status, "return_code": code, "elapsed_seconds": elapsed,
                                  "process_group_empty": cleanup["process_group_empty"], "result_path": result["result_path"]})
        return result

    def _hashes(self):
        return {name: sha256(self.repo / name) for name in SOURCE_FILES}

    def _primary_environment_after(self):
        sys.path.insert(0, str(self.experiments))
        from phase4_shared_memory_prepare import primary_environment
        return primary_environment(self.repo)

    def _finalize_report(self):
        source_after = self._hashes()
        historical_after = {name: sha256(self.repo / name) for name in HISTORICAL_FILES}
        try:
            primary_after = self._primary_environment_after()
            primary_error = None
        except BaseException as error:
            primary_after = None
            primary_error = f"{type(error).__name__}: {error}"
        dependency_after = {name: sha256(self.repo / name) for name in ("pyproject.toml", "uv.lock")}
        source_verified = source_after == self.setup["source_hashes_before"]
        history_unchanged = historical_after == self.setup["historical_files_before"]
        dependency_unchanged = dependency_after == self.setup["primary_dependency_files_before"]
        primary_unchanged = primary_after == self.setup["primary_environment_before"] and primary_error is None
        groups_empty = all(item.get("cleanup", {}).get("process_group_empty") is True
                           for item in self.child_groups)
        if self.active_child:
            groups_empty = False
        if not source_verified:
            self.stop_reason = self.stop_reason or "source_integrity_failure"
        if not history_unchanged:
            self.stop_reason = self.stop_reason or "historical_artifact_changed"
        if not dependency_unchanged or not primary_unchanged:
            self.stop_reason = self.stop_reason or "primary_environment_changed"
        all_stages_passed = all(self.stage_results.get(name, {}).get("status") == "passed"
                                for name in ("kernel_correctness", "training", "reload"))
        elapsed = round(time.monotonic() - self.started, 6)
        terminal_stage = "complete" if all_stages_passed and not self.stop_reason and groups_empty else (
            next((name for name in ("reload", "training", "kernel_correctness")
                  if self.stage_results.get(name, {}).get("status") == "failed"), "preflight_or_monitoring")
        )
        training_result = self.stage_outputs.get("training") or {}
        training_model = read_json(self.run_dir / "training" / "training-model.json") if (self.run_dir / "training" / "training-model.json").is_file() else {}
        training_timing = read_json(self.run_dir / "training" / "training-timing.json") if (self.run_dir / "training" / "training-timing.json").is_file() else {}
        checkpoint = self.run_dir / "training" / "checkpoint"
        metrics_path = checkpoint / "training_metrics.json"
        metrics = read_json(metrics_path) if metrics_path.is_file() else training_timing.get("metrics", {})
        checkpoint_saved = all((checkpoint / name).is_file() for name in ("adapter_model.safetensors", "head.pt", "training_metrics.json"))
        fixture = self.setup["fixture"]
        report = {
            "schema_version": 1,
            "run_id": self.run_id,
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
            "plan_sha256": PLAN_SHA256,
            "model_lock_sha256": MODEL_LOCK_SHA256,
            "resident_vram_gate": "failed_preserved",
            "stop_reason": self.stop_reason,
            "execution": {"started_utc": self.started_utc, "elapsed_seconds": elapsed,
                          "driver_exit_code": 0 if terminal_stage == "complete" else 2,
                          "terminal_stage": terminal_stage, "hard_deadline_monotonic_ns": self.started_ns + int(self.plan["limits"]["wall_seconds"] * 1e9),
                          "deadline_seconds": self.plan["limits"]["wall_seconds"]},
            "setup": {"status": self.setup["status"], "elapsed_seconds": self.setup["setup_elapsed_seconds"],
                      "primary_environment_before": self.setup["primary_environment_before"],
                      "primary_environment_after": primary_after, "primary_environment_error": primary_error,
                      "primary_environment_unchanged": primary_unchanged,
                      "primary_dependency_files_before": self.setup["primary_dependency_files_before"],
                      "primary_dependency_files_after": dependency_after,
                      "primary_dependency_files_unchanged": dependency_unchanged,
                      "overlay": self.setup["overlay"], "gpu_context_initialized": False},
            "stages": self.stage_results,
            "fixture": {"sha256": fixture["fixture_sha256"], "path": fixture["fixture_path"],
                        "record_count": fixture["record_count"], "unique_source_records": fixture["unique_source_records"],
                        "state_tokens_including_state_delimiter": fixture["state_tokens_including_delimiter"],
                        "full_encoded_row_tokens": fixture["full_encoded_row_tokens"],
                        "branch_tokens": fixture["branch_tokens"], "question_id": fixture["question_id"],
                        "question_instructions": fixture["question_instructions"]},
            "kernel_correctness": self.stage_outputs.get("kernel_correctness"),
            "training": {"metrics": metrics, "timing": training_timing,
                         "warm_start_coverage": training_model.get("warm_start_coverage"),
                         "model": training_model, "checkpoint_saved": checkpoint_saved,
                         "checkpoint_path": str(checkpoint)},
            "reload": self.stage_outputs.get("reload"),
            "monitoring": self.monitor.summary(),
            "cleanup": {"reaped": all(item.get("cleanup", {}).get("return_code") is not None
                                         for item in self.child_groups),
                        "process_group_empty": groups_empty, "process_groups": self.child_groups,
                        "windows_sampler": getattr(self, "sampler_stop", None)},
            "source_hashes": {name.rsplit("/", 1)[-1]: digest for name, digest in source_after.items()},
            "source_hashes_by_path": source_after,
            "source_hashes_verified": source_verified,
            "historical_files": {"before": self.setup["historical_files_before"], "after": historical_after,
                                 "unchanged": history_unchanged},
        }
        self.report_path = self.run_dir / "report.json"
        if self.report_path.exists():
            raise RuntimeError(f"refusing to overwrite report: {self.report_path}")
        self.report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report

    def _copy_results(self, report, verdict):
        self.repo_retained_run.parent.mkdir(parents=True, exist_ok=True)
        self.repo_retained_report.mkdir(parents=True, exist_ok=False)
        shutil.copytree(self.run_dir, self.repo_retained_run)
        report_target = self.repo_retained_report / "report.json"
        verdict_target = self.repo_retained_report / "verdict.json"
        shutil.copy2(self.report_path, report_target)
        shutil.copy2(self.run_dir / "gate-verdict.json", verdict_target)
        copied = {"runtime_run": str(self.repo_retained_run), "report_directory": str(self.repo_retained_report),
                  "report_sha256": sha256(self.report_path), "retained_report_sha256": sha256(report_target),
                  "runtime_report_sha256": sha256(self.repo_retained_run / "report.json"),
                  "verdict_sha256": sha256(verdict_target)}
        copied["verified"] = copied["report_sha256"] == copied["retained_report_sha256"] == copied["runtime_report_sha256"]
        if not copied["verified"]:
            raise RuntimeError("final report SHA-256 changed while copying from Linux storage")
        retention = {"schema_version": 1, "run_id": self.run_id, "report": copied,
                     "verdict_status": verdict.get("status"), "recorded_utc": datetime.now(timezone.utc).isoformat()}
        (self.repo_retained_run / "retention.json").write_text(json.dumps(retention, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (self.repo_retained_report / "retention.json").write_text(json.dumps(retention, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return retention

    def execute(self):
        atomic_json(self.run_dir / "runner-stage.json", {"stage": "clock_started_before_gpu_inventory",
                                                         "started_utc": self.started_utc,
                                                         "monotonic_ns": self.started_ns,
                                                         "deadline_seconds": self.plan["limits"]["wall_seconds"]})
        self._record_stage("windows_gpu_inventory", {"status": "running"})
        try:
            first = self.sampler.start(max(0.1, self.deadline - time.monotonic()))
            self.monitor.add_windows_sample(
                first, gpu_query_timeout_seconds=min(
                    self.plan["limits"]["poll_interval_seconds"] * 0.5,
                    max(0.001, self.deadline - time.monotonic())))
            self._record_stage("windows_gpu_inventory", {"status": "sampled",
                                                          "first_sample_monotonic_ns": first[0]})
            if not self.stop_reason:
                kernel_root = self.run_dir / "kernel_correctness"
                kernel_root.mkdir(exist_ok=False)
                kernel_script = self.experiments / "phase4_shared_memory_kernels.py"
                kernel_output = kernel_root / "result.json"
                kernel_stage = kernel_root / "stage.json"
                kernel_observation = kernel_root / "cuda-observation.json"
                command = [str(self.setup["primary_python"]), "-u", str(kernel_script),
                           "--base", "Qwen/Qwen3.5-4B-Base", "--base-revision", "1001bb4d826a52d1f399e183466143f4da7b741b",
                           "--overlay", str(self.overlay), "--output", str(kernel_output),
                           "--stage-file", str(kernel_stage), "--observation", str(kernel_observation)]
                self.run_stage("kernel_correctness", kernel_script, command,
                               [kernel_output], kernel_observation, kernel_stage)

            if not self.stop_reason:
                training_root = self.run_dir / "training"
                proof_root = training_root
                checkpoint = training_root / "checkpoint"
                training_root.mkdir(exist_ok=False)
                train_script = self.experiments / "phase4_shared_memory_train.py"
                train_stage = training_root / "training-stage.json"
                train_observation = training_root / "cuda-observation.json"
                command = [str(self.setup["primary_python"]), "-u", str(train_script),
                           "--phase4-fixture", self.setup["fixture"]["fixture_path"],
                           "--phase4-proof-dir", str(proof_root), "--phase4-observation", str(train_observation),
                           "--phase4-overlay", str(self.overlay),
                           "--data", self.setup["fixture"]["fixture_path"],
                           "--base", "Qwen/Qwen3.5-4B-Base", "--base_revision", "1001bb4d826a52d1f399e183466143f4da7b741b",
                           "--init_from", "jaredpalmer/kev-4b@139fdd94f1b6a6ad80cc15e08fcb99cac885a101",
                           "--head_dim", "256", "--epochs", "1", "--max_steps", "2", "--lr", "2e-5",
                           "--lora", "16", "--lora_targets", "all", "--batch", "1", "--accum", "16",
                           "--device", "cuda", "--dtype", "bf16", "--weights_dtype", "bf16", "--checkpointing", "1",
                           "--max_state", "2048", "--p_none", "0", "--p_none_distract", "0", "--p_distract", "0",
                           "--p_none_pair", "0", "--synthetic_repeat", "1", "--public_frac", "1", "--seed", "0",
                           "--out", str(checkpoint)]
                self.run_stage("training", train_script, command,
                               [training_root / "training-result.json"], train_observation, train_stage)

            if not self.stop_reason:
                reload_root = self.run_dir / "reload"
                reload_root.mkdir(exist_ok=False)
                reload_script = self.experiments / "phase4_shared_memory_reload.py"
                reload_output = reload_root / "result.json"
                reload_stage = reload_root / "stage.json"
                reload_observation = reload_root / "cuda-observation.json"
                command = [str(self.setup["primary_python"]), "-u", str(reload_script),
                           "--model-lock", str(self.experiments / "model-lock-phase-4.json"),
                           "--fixture", self.setup["fixture"]["fixture_path"],
                           "--proof", str(self.run_dir / "training" / "pre-save-output.pt"),
                           "--training-timing", str(self.run_dir / "training" / "training-timing.json"),
                           "--checkpoint", str(self.run_dir / "training" / "checkpoint"),
                           "--stage-file", str(reload_stage), "--observation", str(reload_observation),
                           "--output", str(reload_output)]
                self.run_stage("reload", reload_script, command, [reload_output], reload_observation, reload_stage)
        except BaseException as error:
            if self.stop_reason is None:
                self.stop_reason = "driver_failure"
            atomic_json(self.run_dir / "driver-failure.json", {"type": type(error).__name__, "error": str(error),
                                                                "stage": read_json(self.run_dir / "runner-stage.json")
                                                                if (self.run_dir / "runner-stage.json").is_file() else None})
        finally:
            if self.active_child:
                try:
                    self._stop_child(self.active_child, hard=self.stop_reason == "wall_deadline")
                except BaseException as error:
                    self.stop_reason = self.stop_reason or "cleanup_failure"
                    atomic_json(self.run_dir / "cleanup-failure.json", {"type": type(error).__name__, "error": str(error)})
                self.active_child = None
            try:
                self.sampler_stop = self.sampler.stop()
            except BaseException as error:
                self.stop_reason = self.stop_reason or "monitoring_failure"
                self.sampler_stop = {"stopped": False, "error": f"{type(error).__name__}: {error}"}
            self.monitor.close()

        report = self._finalize_report()
        gate = self.experiments / "phase4_shared_memory_gate.py"
        gate_command = [str(self.setup["primary_python"]), str(gate), "--plan", str(self.plan_path),
                        "--report", str(self.report_path), "--output", str(self.run_dir / "gate-verdict.json")]
        gate_result = subprocess.run(gate_command, cwd=self.repo, text=True,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if gate_result.stdout:
            (self.run_dir / "gate.log").write_text(gate_result.stdout, encoding="utf-8")
        if gate_result.stderr:
            (self.run_dir / "gate.stderr.log").write_text(gate_result.stderr, encoding="utf-8")
        verdict_document = read_json(self.run_dir / "gate-verdict.json")
        verdict = verdict_document.get("verdict", {})
        retention = self._copy_results(report, verdict)
        print(json.dumps({"run_id": self.run_id, "status": verdict.get("status"),
                          "stop_reason": self.stop_reason, "elapsed_seconds": report["execution"]["elapsed_seconds"],
                          "gate_failure_count": verdict.get("failure_count"), "retention": retention}, indent=2, sort_keys=True))
        return 0 if gate_result.returncode == 0 else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="CPU-prepared run directory under the WSL Linux filesystem")
    args = parser.parse_args()
    if not args.run_dir.is_absolute() or args.run_dir.as_posix().startswith("/mnt/"):
        parser.error("--run-dir must be an absolute path in the WSL Linux filesystem")
    runner = Runner(args.repo, args.run_dir)
    return runner.execute()


if __name__ == "__main__":
    raise SystemExit(main())
