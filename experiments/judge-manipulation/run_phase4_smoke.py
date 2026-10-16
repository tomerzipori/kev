"""Run the locked Kev-4B Windows/WSL feasibility smoke and keep all runtime files ignored."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

HF_CACHE = Path(os.environ.get("JUDGE_MANIPULATION_PHASE4_HF_HOME", str(Path.home() / ".cache/huggingface-phase4"))).expanduser().resolve()
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE / "hub")
HF_CACHE.mkdir(parents=True, exist_ok=True)

from kev.suite import digest


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "experiments/judge-manipulation"
LOCK_PATH = EXPERIMENTS / "model-lock-phase-4.json"
TRAIN_WRAPPER = EXPERIMENTS / "phase4_train_instrumented.py"
RELOAD_CHECK = EXPERIMENTS / "verify_phase4_reload.py"


def memory_info():
    values = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.strip().split()[0]) * 1024
    return {key: values[key] for key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree") if key in values}


def gpu_query():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=timestamp,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True, capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(f"nvidia-smi query failed: {result.stderr.strip()}")
    return result.stdout.strip()


def verify_base_weight_files(lock, out_dir):
    from huggingface_hub import hf_hub_download

    metadata = lock["base"]["weight_file_metadata"]
    if not metadata:
        raise RuntimeError("the pinned Qwen base has no safetensors metadata")
    verified = {}
    actual_files = {}
    for index, (name, info) in enumerate(sorted(metadata.items()), start=1):
        print(f"Verifying pinned base file {index}/{len(metadata)}: {name}", flush=True)
        path = Path(hf_hub_download(repo_id=lock["base"]["repo"], filename=name, revision=lock["base"]["revision"]))
        actual = digest(path)
        expected = info.get("lfs_sha256")
        if expected and actual != expected:
            raise RuntimeError(f"base file SHA-256 mismatch for {name}: got {actual}, Hub metadata says {expected}")
        matches = bool(expected and actual == expected)
        verified[name] = {"sha256": actual, "size": path.stat().st_size, "matches_hub_lfs": matches}
        actual_files[name] = {"path": str(path), "sha256": actual, "size": path.stat().st_size}
    (out_dir / "base-weight-verification.json").write_text(json.dumps({"files": verified, "cache_files": actual_files}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return verified


def build_fixture(lock, out_dir, count=32):
    from transformers import AutoTokenizer
    from kev.data import load_records, materialize
    from kev.model import encode, training_context, user_tokens

    base = lock["base"]
    tokenizer = AutoTokenizer.from_pretrained(base["repo"], revision=base["revision"])
    target_user_tokens = 2047
    long_text = "This response explains the recommendation and the evidence behind it in ordinary language. " * 4096
    encoded = tokenizer(long_text, add_special_tokens=False, return_offsets_mapping=True)
    cutoff = encoded["offset_mapping"][target_user_tokens - 1][1]
    state = long_text[:cutoff]
    if len(user_tokens(tokenizer, state)) != target_user_tokens:
        raise RuntimeError("could not construct a state with exactly 2,047 user tokens")
    registration = json.loads((EXPERIMENTS / "registration.json").read_text(encoding="utf-8"))
    record_spec = registration["record"]
    state_sha = hashlib.sha256(state.encode("utf-8")).hexdigest()
    request = {
        "state": state,
        "questions": {
            record_spec["question_id"]: {
                "type": record_spec["type"],
                "instructions": record_spec["instructions"],
                "label": False,
            },
        },
        "_meta": {
            "id": "phase4-hardware-probe-neutral-long-row",
            "group_id": "phase4-hardware-probe-neutral-long-row",
            "source": "phase4_hardware_probe",
            "variant": "synthetic-neutral-long-row",
            "split": "hardware_probe",
            "label_evidence": "Constructed ordinary explanatory response; no instruction or evaluation-control attempt is present.",
            "lineage": {"purpose": "bounded hardware feasibility only", "repeated_copies": count},
            "text_sha256": state_sha,
        },
    }
    fixture = out_dir / "long-state-2048.jsonl"
    with fixture.open("w", encoding="utf-8", newline="\n") as stream:
        for _ in range(count):
            stream.write(json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n")
    record = load_records(fixture)[0]
    context = training_context(2048)
    enc = encode(tokenizer, materialize(record), max_state=context["max_state"], max_branch=context["max_branch"], strict=True)
    if enc["state_tokens"] != 2048:
        raise RuntimeError(f"fixture state is {enc['state_tokens']} tokens including <state>, expected 2048")
    info = {
        "status": "created",
        "purpose": "bounded hardware probe only; repeated identical synthetic rows are not a training corpus",
        "record_count": count,
        "unique_source_records": 1,
        "state_tokens_including_state_delimiter": enc["state_tokens"],
        "branch_tokens": len(enc["ids"]) - enc["state_tokens"],
        "full_encoded_row_tokens": len(enc["ids"]),
        "max_state_tokens": 2048,
        "max_branch_tokens": training_context(2048)["max_branch"],
        "max_packed_tokens": training_context(2048)["max_packed"],
        "question_id": record_spec["question_id"],
        "question_instructions": record_spec["instructions"],
        "raw_state_sha256": state_sha,
        "fixture_sha256": digest(fixture),
        "tokenizer_repo": base["repo"],
        "tokenizer_revision": base["revision"],
        "fixture_path": str(fixture.resolve()),
    }
    (out_dir / "fixture.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return info


def parse_process_memory(pid):
    path = Path(f"/proc/{pid}/status")
    if not path.exists():
        return {"rss_bytes": None, "peak_rss_bytes": None}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            key, value = line.split(":", 1)
            values[key] = int(value.strip().split()[0]) * 1024
    return {"rss_bytes": values.get("VmRSS"), "peak_rss_bytes": values.get("VmHWM")}


def start_monitor(pid, out_dir, stop_event, phase, append=False):
    mode = "a" if append else "w"
    gpu_file = (out_dir / "nvidia-smi.csv").open(mode, encoding="utf-8", newline="")
    ram_file = (out_dir / "wsl-memory.csv").open(mode, encoding="utf-8", newline="")
    gpu_writer = csv.writer(gpu_file)
    ram_writer = csv.writer(ram_file)
    if not append:
        gpu_writer.writerow(["utc", "phase", "gpu_timestamp", "name", "memory_total_mib", "memory_used_mib", "memory_free_mib", "gpu_util_percent"])
        ram_writer.writerow(["utc", "phase", "mem_total_bytes", "mem_available_bytes", "swap_total_bytes", "swap_free_bytes", "process_rss_bytes", "process_peak_rss_bytes"])

    def poll():
        while not stop_event.is_set():
            now = datetime.now(timezone.utc).isoformat()
            try:
                line = gpu_query()
                for row in line.splitlines():
                    gpu_writer.writerow([now, phase, *[item.strip() for item in row.split(",")]])
                mem = memory_info()
                proc = parse_process_memory(pid)
                ram_writer.writerow([now, phase, mem.get("MemTotal"), mem.get("MemAvailable"), mem.get("SwapTotal"), mem.get("SwapFree"), proc["rss_bytes"], proc["peak_rss_bytes"]])
                gpu_file.flush(); ram_file.flush()
            except (OSError, RuntimeError, ValueError):
                pass
            stop_event.wait(1)

    thread = threading.Thread(target=poll, name="phase4-resource-monitor", daemon=True)
    thread.start()
    return thread, gpu_file, ram_file


def env_inventory():
    # CUDA probing runs in a short-lived child so its CUDA context is gone before
    # the measured training or reload process starts.
    code = (
        "import json,torch; "
        "print(json.dumps({'torch':torch.__version__,'torch_cuda_build':torch.version.cuda,"
        "'cuda_available':torch.cuda.is_available(),'cuda_device_count':torch.cuda.device_count(),"
        "'cuda_device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"
        "'cuda_total_memory_bytes':torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,"
        "'bf16_supported':torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False}))"
    )
    probe = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    if probe.returncode:
        raise RuntimeError(f"short-lived CUDA inventory failed: {probe.stderr.strip()}")
    uv_path = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
    uv_version = subprocess.run([uv_path, "--version"], text=True, capture_output=True)
    if uv_version.returncode:
        raise RuntimeError(f"could not read pinned uv version: {uv_version.stderr.strip()}")
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "uv": uv_version.stdout.strip(),
        "python_packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "huggingface-hub")},
        "cuda_probe": json.loads(probe.stdout),
        "nvidia_smi": gpu_query(),
        "wsl_memory": memory_info(),
    }


def classification(log, return_code, stage_record=None):
    lower = log.lower()
    is_oom = "out of memory" in lower or "torch.cuda.outofmemoryerror" in lower or "torch.outofmemoryerror" in lower
    if is_oom:
        stage = stage_record.get("stage", "") if stage_record else ""
        if stage in {"model_load", "warm_start"}:
            return "cuda_oom_model_load"
        for prefix, label in (("forward", "cuda_oom_forward"), ("backward", "cuda_oom_backward"),
                              ("gradient_clip", "cuda_oom_gradient_clip"), ("optimizer_step", "cuda_oom_optimizer"),
                              ("final_proof_forward", "cuda_oom_final_proof_forward"),
                              ("checkpoint_save", "cuda_oom_checkpoint_save")):
            if stage.startswith(prefix):
                return label
        return "cuda_oom_unclassified_stage"
    if "modulenotfounderror" in lower or "no module named" in lower or "importerror" in lower:
        return "dependency_failure"
    if "huggingface" in lower and ("connectionerror" in lower or "http error" in lower or "401 client error" in lower):
        return "model_download_failure"
    if return_code:
        return "training_failure" if "trainable params=" in log else "model_load_or_warm_start_failure"
    return "passed"


def write_reports(out_dir, report_dir, report):
    runtime_path = out_dir / "report-phase-4.json"
    tracked_path = report_dir / "report-phase-4.json"
    tracked_md = report_dir / "report-phase-4.md"
    if tracked_path.exists() or tracked_md.exists():
        raise RuntimeError("refusing to overwrite an existing Phase 4 report artifact")
    runtime_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tracked_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Phase 4 Windows feasibility report",
        "",
        f"Status: **{report['status']}** (`{report['classification']}`).",
        "",
        "The run uses the verified Kev-4B parent and Qwen3.5-4B-Base revisions recorded in `model-lock-phase-4.json`. "
        "It runs the ordinary PyTorch trainer with a BF16 backbone, BF16 autocast, rank-16 LoRA, FP32 adapter/head, "
        "batch 1, accumulation 16, gradient checkpointing, and a 2,048-token state including `<state>`.",
        "",
        "This is a bounded hardware probe using repeated copies of one tokenizer-verified synthetic row. It does not build a corpus or establish detector quality.",
        "",
        "## Execution outcome and memory gate",
        "",
        "```json",
        json.dumps({"execution_outcome": report.get("execution_outcome"), "memory_gate": report.get("memory_gate")}, indent=2, sort_keys=True),
        "```",
        "",
        "## Measurements",
        "",
        "```json",
        json.dumps(report.get("measurements", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## Runtime files",
        "",
        f"Logs, the fixture, and trainer output are under `{out_dir}` (ignored by Git). Model files are resolved in the external Hugging Face cache; their paths and hashes are recorded in the runtime evidence.",
        "",
    ]
    tracked_md.write_text("\n".join(lines), encoding="utf-8")


def resource_summary(out_dir):
    gpu_rows = list(csv.DictReader((out_dir / "nvidia-smi.csv").open(encoding="utf-8", newline="")))
    ram_rows = list(csv.DictReader((out_dir / "wsl-memory.csv").open(encoding="utf-8", newline="")))

    def integer(rows, key):
        return [int(row[key]) for row in rows if row.get(key) not in (None, "", "N/A")]

    used = integer(gpu_rows, "memory_used_mib")
    free = integer(gpu_rows, "memory_free_mib")
    available = integer(ram_rows, "mem_available_bytes")
    rss = integer(ram_rows, "process_rss_bytes")
    peak_rss = integer(ram_rows, "process_peak_rss_bytes")
    return {
        "gpu_sample_count": len(gpu_rows),
        "gpu_peak_used_mib": max(used) if used else None,
        "gpu_min_free_mib": min(free) if free else None,
        "wsl_mem_available_min_bytes": min(available) if available else None,
        "monitored_process_peak_rss_bytes": max(peak_rss or rss) if (peak_rss or rss) else None,
        "monitor_phases": sorted({row.get("phase") for row in gpu_rows if row.get("phase")}),
    }


def memory_budget_assessment(report, out_dir):
    """Keep successful execution separate from the registered physical-memory gate."""
    environment = report.get("environment_before_training") or report.get("environment_before_download") or {}
    probe = environment.get("cuda_probe", {})
    physical_bytes = probe.get("cuda_total_memory_bytes")
    if not isinstance(physical_bytes, int) or physical_bytes <= 0:
        return {
            "status": "not_assessed",
            "reason": "CUDA physical memory capacity was not recorded",
            "physical_device_bytes": physical_bytes,
            "budget_bytes": None,
            "peak_allocated_bytes": None,
            "peak_sources": [],
        }

    budget_bytes = min(8 * 1024**3, physical_bytes)
    evidence = []

    def collect(value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                item_path = f"{path}.{key}"
                if key in {"peak_allocated_bytes", "peak_allocated_bytes_so_far", "reload_peak_allocated_bytes"}:
                    if isinstance(item, int) and item >= 0:
                        evidence.append({"source": item_path, "bytes": item})
                collect(item, item_path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                collect(item, f"{path}[{index}]")

    proof_dir = out_dir / "proof"
    # Include embedded evidence so this assessment can be replayed from the
    # retained report even when ignored runtime proof files are unavailable.
    collect(report, "report")
    for name in ("in_training_model.json", "training-failure-memory.json", "save-reload-verification.json"):
        path = proof_dir / name
        if path.is_file():
            collect(json.loads(path.read_text(encoding="utf-8")), str(path))

    peak = max((item["bytes"] for item in evidence), default=None)
    if peak is None:
        status = "not_assessed"
        reason = "no CUDA peak-allocation evidence was recorded"
    elif peak > budget_bytes:
        status = "failed_memory_budget"
        reason = "peak CUDA allocation exceeded min(8 GiB, physical device memory)"
    else:
        status = "within_limit"
        reason = None
    return {
        "status": status,
        "reason": reason,
        "budget_bytes": budget_bytes,
        "budget_rule": "min(8 * 1024^3, actual CUDA physical device memory)",
        "physical_device_bytes": physical_bytes,
        "peak_allocated_bytes": peak,
        "peak_sources": evidence,
        "exceeded_by_bytes": max(0, peak - budget_bytes) if peak is not None else None,
    }


def reload_oom_classification(log, stage_record):
    lower = log.lower()
    if not any(marker in lower for marker in ("out of memory", "torch.cuda.outofmemoryerror", "torch.outofmemoryerror")):
        return "save_reload_failure"
    stage = stage_record.get("stage", "") if stage_record else ""
    if stage in {"reload_model_load", "reload_warm_start"}:
        return "cuda_oom_reload_model_load"
    if stage.startswith("reload_forward"):
        return "cuda_oom_reload_forward"
    return "cuda_oom_reload_unclassified_stage"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "runs/judge-manipulation-phase4/windows-smoke")
    parser.add_argument("--report-dir", type=Path, default=EXPERIMENTS)
    parser.add_argument("--records", type=int, default=32)
    parser.add_argument("--optimizer-steps", type=int, default=2)
    args = parser.parse_args()
    if args.records != 32 or args.optimizer_steps != 2:
        raise RuntimeError("the reviewed hardware probe is fixed at 32 repeated rows and exactly two optimizer steps")
    os.chdir(ROOT)
    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    out_dir = out_dir.resolve()
    relative_out = out_dir.relative_to(ROOT)
    report_dir = args.report_dir if args.report_dir.is_absolute() else ROOT / args.report_dir
    report_dir = report_dir.resolve()
    try:
        report_dir.relative_to(EXPERIMENTS)
    except ValueError:
        raise RuntimeError(f"report directory must stay within {EXPERIMENTS}") from None
    if (report_dir / "report-phase-4.json").exists() or (report_dir / "report-phase-4.md").exists():
        raise RuntimeError("refusing to overwrite existing Phase 4 reports; choose a fresh --report-dir")
    ignored = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-q", str(relative_out)], check=False).returncode == 0
    if not ignored:
        raise RuntimeError(f"runtime directory is not ignored by Git: {relative_out}")
    if out_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing run directory: {out_dir}")
    out_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    overall_started = time.monotonic()
    started_utc = datetime.now(timezone.utc)
    started_local = datetime.now(ZoneInfo("Asia/Jerusalem"))
    report = {
        "schema_version": 1,
        "recorded_on": started_local.date().isoformat(),
        "report_timezone": "Asia/Jerusalem",
        "started_utc": started_utc.isoformat(),
        "started_local": started_local.isoformat(),
        "reference_host": {"description": "original Windows desktop target, not inferred for portable reruns",
                           "reported_physical_ram_gib": 32,
                           "wslconfig_target": "memory=20GB, swap=8GB, processors=12"},
        "runtime_directory": str(out_dir),
        "report_directory": str(report_dir),
        "huggingface_cache": str(HF_CACHE),
        "runtime_directory_git_ignored": ignored,
        "corpus_built": False,
        "baseline_evaluation_run": False,
        "locked_test_opened": False,
        "status": "running",
        "classification": "not_started",
    }
    try:
        if not LOCK_PATH.is_file():
            raise RuntimeError("missing model-lock-phase-4.json; run model_lock_phase4.py first")
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        if lock.get("status") != "locked":
            raise RuntimeError("model lock is not marked locked")
        report["model_lock_path"] = str(LOCK_PATH)
        report["model_lock_sha256_before_weight_verification"] = digest(LOCK_PATH)
        report["environment_before_download"] = env_inventory()
        (out_dir / "environment-before.json").write_text(json.dumps(report["environment_before_download"], indent=2, sort_keys=True) + "\n", encoding="utf-8")

        base_download_started = time.monotonic()
        report["base_weight_verification"] = verify_base_weight_files(lock, out_dir)
        report["base_weight_verification_elapsed_seconds"] = round(time.monotonic() - base_download_started, 3)
        report["model_lock_path"] = str(LOCK_PATH)
        report["model_lock_sha256"] = digest(LOCK_PATH)
        if report["model_lock_sha256"] != report["model_lock_sha256_before_weight_verification"]:
            raise RuntimeError("model-lock-phase-4.json changed during base weight verification")
        report["environment_before_training"] = env_inventory()
        (out_dir / "environment-before-training.json").write_text(json.dumps(report["environment_before_training"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report["fixture"] = build_fixture(lock, out_dir, args.records)
        proof_dir = out_dir / "proof"
        proof_dir.mkdir()
        command = [
            sys.executable, str(TRAIN_WRAPPER),
            "--phase4-fixture", str(out_dir / "long-state-2048.jsonl"),
            "--phase4-proof-dir", str(proof_dir),
            "--data", str(out_dir / "long-state-2048.jsonl"),
            "--base", lock["base"]["repo"],
            "--base_revision", lock["base"]["revision"],
            "--init_from", f"{lock['parent']['repo']}@{lock['parent']['weights_revision']}",
            "--epochs", "1", "--max_steps", str(args.optimizer_steps),
            "--lr", "2e-5", "--lora", "16", "--lora_targets", "all",
            "--batch", "1", "--accum", "16", "--device", "cuda",
            "--dtype", "bf16", "--weights_dtype", "bf16", "--checkpointing", "1",
            "--max_state", "2048", "--p_none", "0", "--p_none_distract", "0",
            "--p_distract", "0", "--p_none_pair", "0", "--synthetic_repeat", "1",
            "--public_frac", "1", "--seed", "0",
            "--out", str((out_dir / "training-output").resolve()),
        ]
        report["training_command"] = command
        (out_dir / "training-command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
        log_path = out_dir / "training.log"
        report["gpu_before_training"] = gpu_query()
        report["gpu_processes_before_training"] = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            text=True, capture_output=True,
        ).stdout.strip().splitlines()
        stop_monitor = threading.Event()
        training_started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log_stream:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log_stream, stderr=subprocess.STDOUT, text=True)
            thread, gpu_file, ram_file = start_monitor(process.pid, out_dir, stop_monitor, "training")
            return_code = process.wait()
            stop_monitor.set()
            thread.join(timeout=5)
            gpu_file.close(); ram_file.close()
        report["training_elapsed_seconds"] = round(time.monotonic() - training_started, 3)
        log = log_path.read_text(encoding="utf-8", errors="replace")
        report["training_return_code"] = return_code
        stage_file = proof_dir / "stage.json"
        stage_record = json.loads(stage_file.read_text(encoding="utf-8")) if stage_file.is_file() else None
        report["training_stage_at_exit"] = stage_record
        training_failure_memory = proof_dir / "training-failure-memory.json"
        if training_failure_memory.is_file():
            report["training_failure_memory"] = json.loads(training_failure_memory.read_text(encoding="utf-8"))
        report["classification"] = classification(log, return_code, stage_record)
        report["training_loaded_model"] = "trainable params=" in log
        report["training_error_tail"] = log.splitlines()[-35:]
        report["gpu_after_training"] = gpu_query()
        training_metrics = out_dir / "training-output/training_metrics.json"
        if training_metrics.is_file():
            report["training_metrics"] = json.loads(training_metrics.read_text(encoding="utf-8"))
        if return_code == 0:
            proof_command = [
                sys.executable, str(RELOAD_CHECK),
                "--model-lock", str(LOCK_PATH),
                "--fixture", str(out_dir / "long-state-2048.jsonl"),
                "--proof", str(proof_dir / "pre-save-output.pt"),
                "--checkpoint", str(out_dir / "training-output"),
                "--stage-file", str(proof_dir / "reload-stage.json"),
                "--output", str(proof_dir / "save-reload-verification.json"),
            ]
            (out_dir / "reload-command.json").write_text(json.dumps(proof_command, indent=2) + "\n", encoding="utf-8")
            reload_log = out_dir / "save-reload.log"
            reload_stop = threading.Event()
            reload_started = time.monotonic()
            with reload_log.open("w", encoding="utf-8") as stream:
                checked = subprocess.Popen(proof_command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, text=True)
                reload_thread, reload_gpu_file, reload_ram_file = start_monitor(checked.pid, out_dir, reload_stop, "reload", append=True)
                reload_return_code = checked.wait()
                reload_stop.set()
                reload_thread.join(timeout=5)
                reload_gpu_file.close(); reload_ram_file.close()
            report["save_reload_elapsed_seconds"] = round(time.monotonic() - reload_started, 3)
            report["save_reload_return_code"] = reload_return_code
            report["save_reload_log"] = str(reload_log)
            reload_stage_file = proof_dir / "reload-stage.json"
            reload_stage = json.loads(reload_stage_file.read_text(encoding="utf-8")) if reload_stage_file.is_file() else None
            report["save_reload_stage_at_exit"] = reload_stage
            if reload_return_code == 0:
                report["classification"] = "passed"
                report["save_reload_verification"] = json.loads((proof_dir / "save-reload-verification.json").read_text(encoding="utf-8"))
            else:
                reload_log_text = reload_log.read_text(encoding="utf-8", errors="replace")
                report["classification"] = reload_oom_classification(reload_log_text, reload_stage)
                report["save_reload_error_tail"] = reload_log_text.splitlines()[-35:]
        report["execution_outcome"] = {
            "status": "passed" if report["classification"] == "passed" else "failed",
            "classification": report["classification"],
            "training_return_code": report.get("training_return_code"),
            "save_reload_return_code": report.get("save_reload_return_code"),
            "training_terminal_stage": (report.get("training_stage_at_exit") or {}).get("stage"),
        }
        report["memory_gate"] = memory_budget_assessment(report, out_dir)
        if report["memory_gate"]["status"] == "failed_memory_budget":
            report["status"] = "failed"
            report["classification"] = "memory_budget_exceeded"
        elif (report["execution_outcome"]["status"] == "passed"
              and report["memory_gate"]["status"] == "within_limit"):
            report["status"] = "passed"
        elif (report["execution_outcome"]["status"] == "passed"
              and report["memory_gate"]["status"] == "not_assessed"):
            report["status"] = "failed"
            report["classification"] = "memory_gate_not_assessed"
        else:
            report["status"] = "failed"
        report["hardware_feasibility_status"] = report["memory_gate"]["status"]
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["finished_local"] = datetime.now(ZoneInfo("Asia/Jerusalem")).isoformat()
        measurements = {
            "runtime_environment_before_training": report.get("environment_before_training"),
            "gpu_before_training": report.get("gpu_before_training"),
            "gpu_after_training": report.get("gpu_after_training"),
            "fixture": report.get("fixture"),
            "training_metrics": report.get("training_metrics"),
            "training_failure_memory": report.get("training_failure_memory"),
            "training_elapsed_seconds": report.get("training_elapsed_seconds"),
            "save_reload_elapsed_seconds": report.get("save_reload_elapsed_seconds"),
            "base_weight_verification_elapsed_seconds": report.get("base_weight_verification_elapsed_seconds"),
            "in_training_model": json.loads((proof_dir / "in_training_model.json").read_text(encoding="utf-8")) if (proof_dir / "in_training_model.json").is_file() else None,
            "save_reload": report.get("save_reload_verification"),
            "resource_summary": resource_summary(out_dir),
            "resource_samples": {"gpu_csv": str(out_dir / "nvidia-smi.csv"), "wsl_memory_csv": str(out_dir / "wsl-memory.csv")},
        }
        report["measurements"] = measurements
        report["overall_elapsed_seconds"] = round(time.monotonic() - overall_started, 3)
        report["report_json_path"] = str(report_dir / "report-phase-4.json")
        report["report_markdown_path"] = str(report_dir / "report-phase-4.md")
        write_reports(out_dir, report_dir, report)
        print(json.dumps({"status": report["status"], "classification": report["classification"],
                          "report": str(report_dir / "report-phase-4.json"), "runtime": str(out_dir)}, indent=2))
        return 0 if report["status"] == "passed" else 2
    except Exception as error:
        report["status"] = "failed"
        report["classification"] = "preflight_or_download_failure"
        report["error"] = f"{type(error).__name__}: {error}"
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["finished_local"] = datetime.now(ZoneInfo("Asia/Jerusalem")).isoformat()
        report["measurements"] = {"environment": report.get("environment_before_download"), "gpu": report.get("gpu_before_training")}
        if "overall_elapsed_seconds" not in report:
            report["overall_elapsed_seconds"] = round(time.monotonic() - overall_started, 3)
        report["report_json_path"] = str(report_dir / "report-phase-4.json")
        report["report_markdown_path"] = str(report_dir / "report-phase-4.md")
        write_reports(out_dir, report_dir, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
