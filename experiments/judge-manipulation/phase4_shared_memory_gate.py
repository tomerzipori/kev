"""Evaluate the amended Phase 4 receipt without inferring missing evidence."""

import argparse
import json
import math
from pathlib import Path


PLAN_SHA256 = "0329f3bc4fc6e9779caf1b32b393729b9f36fc6d28de85bb17e7c1b3bae4b48a"
LOCK_SHA256 = "0bdce0513f5d85c09154f6a8808ad973bc2848d679d1144c7093e64dc6d8fb19"
FIXTURE_SHA256 = "ccc4c494c47e2559334b93842ce30c218da0e0fb6234eaf8d111aa2c5c97be20"
EXPECTED_PACKAGES = {
    "torch": "2.8.0+cu128",
    "transformers": "5.17.0",
    "peft": "0.21.0",
    "flash-linear-attention": "0.5.2",
    "triton": "3.4.0",
    "causal-conv1d": "1.7.0",
}
LIMIT_NAMES = (
    "wall_seconds", "poll_interval_seconds", "consecutive_violating_samples",
    "cuda_peak_allocated_bytes_max", "cuda_peak_reserved_bytes_max",
    "windows_adapter_shared_bytes_max", "active_model_process_rss_bytes_max",
    "windows_available_physical_bytes_min", "wsl_memavailable_bytes_min",
    "wsl_swap_used_bytes_max",
)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _object(value):
    return value if isinstance(value, dict) else {}


def _under(path, root):
    path = str(path or "").replace("\\", "/").rstrip("/")
    root = str(root or "").replace("\\", "/").rstrip("/")
    return bool(path and root and path.startswith(root + "/"))


def evaluate(report, plan):
    """Return a fail-closed verdict over the serialized watchdog receipt."""
    failures = []

    def require(condition, path, reason):
        if not condition:
            failures.append({"path": path, "reason": reason})

    if not isinstance(report, dict):
        return {"status": "failed_incomplete", "shared_memory_gate": "failed_incomplete",
                "resident_vram_gate": "failed_preserved", "failure_count": 1,
                "failures": [{"path": "$", "reason": "report root must be a JSON object"}]}
    limits = _object(_object(plan).get("limits"))
    if not isinstance(plan, dict) or any(not _number(limits.get(key)) for key in LIMIT_NAMES):
        return {"status": "failed_incomplete", "shared_memory_gate": "failed_incomplete",
                "resident_vram_gate": "failed_preserved", "failure_count": 1,
                "failures": [{"path": "plan", "reason": "approved plan limits are missing or malformed"}]}

    require(report.get("plan_sha256") == PLAN_SHA256, "plan_sha256", "does not identify the approved plan")
    require(report.get("model_lock_sha256") == LOCK_SHA256, "model_lock_sha256", "does not identify the frozen model lock")
    require(report.get("resident_vram_gate") == "failed_preserved", "resident_vram_gate", "historical resident-VRAM failure was not preserved")
    require(report.get("stop_reason") is None, "stop_reason", "the benchmark stopped before completing")

    execution = _object(report.get("execution"))
    elapsed = execution.get("elapsed_seconds")
    require(_number(elapsed) and 0 <= elapsed <= limits["wall_seconds"],
            "execution.elapsed_seconds", "missing or outside the hard wall deadline")
    require(execution.get("driver_exit_code") == 0, "execution.driver_exit_code", "the staged benchmark driver did not exit cleanly")
    require(execution.get("terminal_stage") == "complete", "execution.terminal_stage", "the complete stage was not reached")

    expected_stages = ("kernel_correctness", "training", "reload")
    stages = _object(report.get("stages"))
    kernel_stage = _object(stages.get("kernel_correctness"))
    train_stage = _object(stages.get("training"))
    reload_stage = _object(stages.get("reload"))
    for name in expected_stages:
        stage = _object(stages.get(name))
        require(stage.get("status") == "passed", f"stages.{name}.status", "required stage did not pass")
    require(kernel_stage.get("separate_process") is True,
            "stages.kernel_correctness.separate_process", "kernel check did not run in its own process")
    require(train_stage.get("fresh_process") is True,
            "stages.training.fresh_process", "training did not run in a fresh process")
    require(reload_stage.get("fresh_process") is True,
            "stages.reload.fresh_process", "reload did not run in a fresh process")

    fixture = _object(report.get("fixture"))
    require(fixture.get("sha256") == FIXTURE_SHA256, "fixture.sha256", "fixture differs from the approved hardware fixture")
    require(fixture.get("record_count") == 32 and fixture.get("unique_source_records") == 1,
            "fixture.record_count", "fixture must contain 32 copies of one neutral record")
    require(fixture.get("state_tokens_including_state_delimiter") == 2048,
            "fixture.state_tokens_including_state_delimiter", "state token count differs from the registration")
    require(fixture.get("full_encoded_row_tokens") == 2099 and fixture.get("branch_tokens") == 51,
            "fixture.full_encoded_row_tokens", "question branch or full row token count differs from the registration")

    training = _object(report.get("training"))
    metrics = _object(training.get("metrics"))
    require(metrics.get("optimizer_steps") == 2, "training.metrics.optimizer_steps", "exactly two AdamW updates are required")
    require(metrics.get("records_seen") == 32, "training.metrics.records_seen", "all 32 registered microbatches must train")
    step_seconds = metrics.get("step_seconds")
    require(isinstance(step_seconds, list) and len(step_seconds) == 2 and all(_number(x) and x > 0 for x in step_seconds),
            "training.metrics.step_seconds", "both optimizer update intervals must be present and positive")
    grad_norms = metrics.get("grad_norm", [])
    if not isinstance(grad_norms, list):
        grad_norms = []
        require(False, "training.metrics.grad_norm", "gradient norm evidence must be a list")
    grad_steps = 0
    for epoch in grad_norms:
        if not isinstance(epoch, dict) or not isinstance(epoch.get("steps"), int) or isinstance(epoch.get("steps"), bool):
            require(False, "training.metrics.grad_norm", "each gradient summary must report an integer step count")
        else:
            grad_steps += epoch["steps"]
    require(grad_steps == 2, "training.metrics.grad_norm", "two optimizer updates need two reported gradient norms")
    require(all(_number(epoch.get(key)) and epoch[key] > 0
                for epoch in grad_norms if isinstance(epoch, dict) for key in ("mean", "max")),
            "training.metrics.grad_norm", "gradient norms must be finite and nonzero")
    timing = _object(training.get("timing"))
    update_norms = timing.get("per_update_gradient_norms")
    require(isinstance(update_norms, list) and len(update_norms) == 2
            and all(_number(value) and value > 0 for value in update_norms),
            "training.timing.per_update_gradient_norms", "each optimizer update needs a finite nonzero gradient norm")
    for field in ("optimizer_operation_seconds", "optimizer_window_seconds"):
        values = timing.get(field)
        require(isinstance(values, list) and len(values) == 2 and all(_number(value) and value > 0 for value in values),
                f"training.timing.{field}", "both measured optimizer intervals must be finite and positive")
    require(timing.get("microbatches") == 32 and timing.get("optimizer_updates") == 2,
            "training.timing", "timing proof must cover 32 microbatches and two completed updates")
    warm_start = _object(training.get("warm_start_coverage"))
    require(_number(warm_start.get("tensors")) and warm_start.get("tensors") > 0,
            "training.warm_start_coverage", "canonical parent adapter and head warm start is missing")
    model = _object(training.get("model"))
    recipe = _object(model.get("recipe_arguments"))
    require(model.get("base") == "Qwen/Qwen3.5-4B-Base"
            and model.get("base_revision") == "1001bb4d826a52d1f399e183466143f4da7b741b",
            "training.model.base", "training must use the registered base and immutable revision")
    require(str(model.get("init_from", "")).endswith("@139fdd94f1b6a6ad80cc15e08fcb99cac885a101"),
            "training.model.init_from", "strict warm start must use the registered parent revision")
    require(model.get("lora_ranks") == {"default": 16} and model.get("lora_targets_argument") == "all",
            "training.model.lora", "rank-16 all-target LoRA recipe changed")
    require(model.get("option_isolation") is False and model.get("special_embeddings") is False
            and model.get("full_weight_training") is False,
            "training.model.architecture", "registered LoRA-only architecture settings changed")
    expected_recipe = {"--head_dim": "256", "--batch": "1", "--accum": "16", "--checkpointing": "1",
                       "--epochs": "1",
                       "--lr": "2e-5", "--lora": "16", "--lora_targets": "all", "--dtype": "bf16",
                       "--weights_dtype": "bf16", "--max_state": "2048", "--seed": "0",
                       "--p_none": "0", "--p_none_distract": "0", "--p_distract": "0", "--p_none_pair": "0",
                       "--synthetic_repeat": "1", "--public_frac": "1", "--max_steps": "2"}
    require(all(recipe.get(name) == value for name, value in expected_recipe.items()),
            "training.model.recipe_arguments", "registered training recipe arguments changed or are missing")
    dtype = _object(model.get("parameter_dtype_evidence"))
    frozen_counts = _object(dtype.get("frozen_parameter_dtype_counts"))
    require(_number(frozen_counts.get("bfloat16")) and frozen_counts["bfloat16"] > 0
            and set(frozen_counts) <= {"bfloat16", "float32"},
            "training.model.parameter_dtype_evidence.frozen_parameter_dtype_counts", "frozen backbone was not loaded in BF16")
    require(sorted(_object(dtype.get("lora_trainable_parameter_dtype_counts"))) == ["float32"],
            "training.model.parameter_dtype_evidence.lora_trainable_parameter_dtype_counts", "trainable LoRA parameters must be FP32")
    require(model.get("head_dtypes") == ["float32"], "training.model.head_dtypes", "pointer head must be FP32")
    train_env = _object(model.get("kernel_environment"))
    train_delta = _object(train_env.get("deltanet"))
    train_packages = _object(train_env.get("packages"))
    train_bindings = _object(model.get("kernel_bindings"))
    train_overlay = model.get("overlay_path")
    require(bool(train_env.get("gpu")), "training.model.kernel_environment.gpu", "canonical loaded-model GPU environment is missing")
    require(str(train_delta.get("causal_conv1d_fn", "")).startswith("causal_conv1d."),
            "training.model.kernel_environment.deltanet.causal_conv1d_fn", "Transformers is not bound to the installed causal-conv1d kernel")
    require(str(train_delta.get("torch_chunk_gated_delta_rule", "")).startswith("fla."),
            "training.model.kernel_environment.deltanet.torch_chunk_gated_delta_rule", "Transformers is not bound to the installed FLA kernel")
    for name, prefix in (("causal_conv1d_fn", "causal_conv1d."),
                         ("torch_chunk_gated_delta_rule", "fla.")):
        binding = _object(train_bindings.get(name))
        require(str(binding.get("qualified_name", "")).startswith(prefix),
                f"training.model.kernel_bindings.{name}.qualified_name", "loaded model did not bind the required optional-package implementation")
        require(_under(binding.get("source"), train_overlay),
                f"training.model.kernel_bindings.{name}.source", "loaded model kernel source is outside the registered overlay")
    require(bool(train_overlay), "training.model.overlay_path", "training overlay path is missing")
    for package, version in EXPECTED_PACKAGES.items():
        require(train_packages.get(package) == version, f"training.model.kernel_environment.packages.{package}",
                f"expected pinned package version {version}")
    require(training.get("checkpoint_saved") is True, "training.checkpoint_saved", "checkpoint save did not complete")

    correctness = _object(report.get("kernel_correctness"))
    optimized = _object(correctness.get("optimized_functions"))
    optimized_delta = _object(optimized.get("delta_rule"))
    optimized_conv = _object(optimized.get("causal_conv1d"))
    correctness_packages = _object(correctness.get("package_versions"))
    require(correctness.get("status") == "passed", "kernel_correctness.status", "separate operator check did not pass")
    require(str(optimized_delta.get("module", "")).startswith("fla."),
            "kernel_correctness.optimized_functions.delta_rule", "operator check did not call FLA directly")
    require(str(optimized_conv.get("module", "")).startswith("causal_conv1d."),
            "kernel_correctness.optimized_functions.causal_conv1d", "operator check did not call causal-conv1d directly")
    for package, version in (("flash-linear-attention", "0.5.2"), ("fla-core", "0.5.2"),
                             ("causal-conv1d", "1.7.0"), ("triton", "3.4.0")):
        require(correctness_packages.get(package) == version, f"kernel_correctness.package_versions.{package}",
                f"expected pinned kernel package version {version}")
    imported_paths = _object(correctness.get("imported_paths"))
    correctness_overlay = correctness.get("overlay_path")
    require(_under(imported_paths.get("fla"), correctness_overlay)
            and _under(optimized_delta.get("source"), correctness_overlay),
            "kernel_correctness.optimized_functions.delta_rule.source", "FLA source is not from the registered optional-package import")
    require(_under(imported_paths.get("causal_conv1d"), correctness_overlay)
            and _under(optimized_conv.get("source"), correctness_overlay),
            "kernel_correctness.optimized_functions.causal_conv1d.source", "causal-conv1d source is not from the registered optional-package import")
    require(bool(correctness_overlay), "kernel_correctness.overlay_path", "operator-check overlay path is missing")

    reload = _object(report.get("reload"))
    parity = _object(reload.get("save_reload"))
    require(parity.get("exact_logits") is True and parity.get("max_abs_logit_delta") == 0,
            "reload.save_reload", "fresh unmerged reload logits do not match exactly")
    require(reload.get("temperature_pre_save") == reload.get("temperature_reload") and _number(reload.get("temperature_reload")),
            "reload.temperature_reload", "head temperature changed across save and reload")
    require(reload.get("temperature_meta_pre_save") == reload.get("temperature_pre_save"),
            "reload.temperature_meta_pre_save", "the model head and checkpoint metadata temperatures differed before save")
    eval_reload = _object(reload.get("eval_state_reload"))
    require(reload.get("eval_state_pre_save") == reload.get("eval_state_reload")
            and eval_reload.get("model_training") is False,
            "reload.eval_state_reload", "training/eval state differs across save and reload")
    tf32_before, tf32_after = _object(reload.get("tf32_pre_save")), _object(reload.get("tf32_reload"))
    require(tf32_before == tf32_after and tf32_after.get("matmul") is True and tf32_after.get("cudnn") is True,
            "reload.tf32_reload", "TF32 flags differ from the registered settings")
    env_before, env_after = reload.get("kernel_environment_pre_save"), reload.get("kernel_environment_reload")
    require(isinstance(env_before, dict) and env_before == env_after,
            "reload.kernel_environment_reload", "canonical kernel environment changed across reload")
    require(reload.get("unmerged_lora") is True, "reload.unmerged_lora", "reload must keep the LoRA adapter unmerged")
    require(_number(reload.get("changed_adapter_tensor_count")) and reload.get("changed_adapter_tensor_count") > 0,
            "reload.changed_adapter_tensor_count", "training must change an adapter tensor")
    require(_number(reload.get("changed_head_tensor_count")) and reload.get("changed_head_tensor_count") > 0,
            "reload.changed_head_tensor_count", "training must change a pointer-head tensor")

    monitoring = _object(report.get("monitoring"))
    samples = monitoring.get("samples", [])
    if not isinstance(samples, list):
        samples = []
    require(monitoring.get("status") == "complete", "monitoring.status", "resource monitoring did not complete")
    require(isinstance(samples, list) and len(samples) >= 1, "monitoring.samples", "no resource samples were retained")
    max_gap = monitoring.get("max_gap_seconds")
    require(_number(max_gap) and max_gap < limits["consecutive_violating_samples"],
            "monitoring.max_gap_seconds", "resource sampling missed three consecutive one-second opportunities")
    keys = {
        "windows_adapter_shared_bytes": ("max", limits["windows_adapter_shared_bytes_max"]),
        "active_model_process_rss_bytes": ("max", limits["active_model_process_rss_bytes_max"]),
        "windows_available_physical_bytes": ("min", limits["windows_available_physical_bytes_min"]),
        "wsl_memavailable_bytes": ("min", limits["wsl_memavailable_bytes_min"]),
        "wsl_swap_used_bytes": ("max", limits["wsl_swap_used_bytes_max"]),
        "cuda_peak_allocated_bytes": ("max", limits["cuda_peak_allocated_bytes_max"]),
        "cuda_peak_reserved_bytes": ("max", limits["cuda_peak_reserved_bytes_max"]),
        "gpu_dedicated_used_bytes": ("max", None),
        "gpu_dedicated_free_bytes": ("min", None),
    }
    # The approved monitor and gate share the same three-consecutive-sample
    # stop policy. Isolated threshold crossings remain in the receipt and are
    # counted; they do not create a different, stricter stop rule here.
    maximum_missing_streak = 0
    maximum_violation_streak = 0
    missing_streak = violation_streak = 0
    violation_samples = 0
    missing_samples = 0
    observed_times = []
    required_sample_keys = ("windows_adapter_shared_bytes", "active_model_process_rss_bytes",
                            "windows_available_physical_bytes", "wsl_memavailable_bytes", "wsl_swap_used_bytes",
                            "gpu_dedicated_used_bytes", "gpu_dedicated_free_bytes")
    for sample in samples:
        if not isinstance(sample, dict):
            missing_streak += 1
            violation_streak = 0
            missing_samples += 1
            maximum_missing_streak = max(maximum_missing_streak, missing_streak)
            continue
        sample_missing = False
        sample_violation = False
        sample_kind = sample.get("sample_kind", "resource_poll")
        if sample_kind not in {"resource_poll", "missed_poll"}:
            sample_missing = True
        elif sample_kind == "missed_poll":
            sample_missing = True
        monotonic_ns = sample.get("monotonic_ns")
        if sample_kind == "resource_poll" and _number(monotonic_ns) and monotonic_ns >= 0:
            observed_times.append(monotonic_ns)
        elif sample_kind == "resource_poll":
            sample_missing = True
        if not isinstance(sample.get("cuda_active"), bool):
            sample_missing = True
        for name in required_sample_keys:
            value = sample.get(name)
            if not _number(value) or value < 0:
                sample_missing = True
        for name, (direction, limit) in keys.items():
            if name.startswith("cuda_") and sample.get("cuda_active") is not True:
                continue
            value = sample.get(name)
            if value is None:
                if name.startswith("cuda_"):
                    sample_missing = True
                continue
            if not _number(value) or value < 0:
                sample_missing = True
                continue
            if limit is not None and _number(value):
                ok = value <= limit if direction == "max" else value >= limit
                sample_violation = sample_violation or not ok
        missing_streak = missing_streak + 1 if sample_missing else 0
        violation_streak = violation_streak + 1 if sample_violation else 0
        maximum_missing_streak = max(maximum_missing_streak, missing_streak)
        maximum_violation_streak = max(maximum_violation_streak, violation_streak)
        missing_samples += int(sample_missing)
        violation_samples += int(sample_violation)

    reported_missing_streak = monitoring.get("max_consecutive_missing_samples")
    reported_violation_streak = monitoring.get("max_consecutive_violating_samples")
    reported_violation_samples = monitoring.get("violating_sample_count")
    reported_missing_samples = monitoring.get("missing_sample_count")
    gaps = [(right - left) / 1e9 for left, right in zip(observed_times, observed_times[1:])]
    actual_max_gap = max(gaps, default=None)
    require(len(observed_times) >= 2 and all(right > left for left, right in zip(observed_times, observed_times[1:])),
            "monitoring.samples.monotonic_ns", "at least two strictly increasing actual monotonic timestamps are required")
    require(_number(max_gap) and _number(actual_max_gap) and max_gap >= actual_max_gap
            and actual_max_gap < limits["consecutive_violating_samples"],
            "monitoring.max_gap_seconds", "reported poll gaps are missing, undercounted, or span three consecutive polling opportunities")
    require(_number(reported_missing_streak) and reported_missing_streak >= maximum_missing_streak
            and reported_missing_streak < limits["consecutive_violating_samples"],
            "monitoring.max_consecutive_missing_samples", "critical monitoring was missing for three consecutive samples or the report undercounts gaps")
    require(_number(reported_violation_streak) and reported_violation_streak >= maximum_violation_streak
            and reported_violation_streak < limits["consecutive_violating_samples"],
            "monitoring.max_consecutive_violating_samples", "three consecutive resource-limit samples should have stopped the run")
    require(_number(reported_violation_samples) and reported_violation_samples >= violation_samples,
            "monitoring.violating_sample_count", "resource-limit violations were omitted from the report")
    require(_number(reported_missing_samples) and reported_missing_samples >= missing_samples,
            "monitoring.missing_sample_count", "samples with missing critical measurements were omitted from the report")

    cleanup = _object(report.get("cleanup"))
    require(cleanup.get("reaped") is True and cleanup.get("process_group_empty") is True,
            "cleanup", "benchmark process group was not confirmed empty")
    sampler = _object(cleanup.get("windows_sampler"))
    require(sampler.get("stopped") is True and sampler.get("reader_stopped") is True
            and sampler.get("exit_code") == 0,
            "cleanup.windows_sampler", "Windows counter sampler and its reader did not stop cleanly")

    source_hashes = report.get("source_hashes", {})
    require(report.get("source_hashes_verified") is True and isinstance(source_hashes, dict) and len(source_hashes) >= 4,
            "source_hashes", "source and plan hashes were not recorded and verified")
    history = _object(report.get("historical_files"))
    require(history.get("before") == history.get("after") and history.get("unchanged") is True,
            "historical_files", "a frozen historical artifact changed during the run")

    return {
        "status": "passed" if not failures else "failed_incomplete",
        "shared_memory_gate": "passed" if not failures else "failed_incomplete",
        "resident_vram_gate": "failed_preserved",
        "resource_threshold_crossings_below_stop_streak": violation_samples,
        "stop_policy": "three consecutive violating samples; isolated crossings remain recorded in monitoring.samples",
        "failure_count": len(failures),
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error(f"refusing to overwrite existing verdict: {args.output}")
    try:
        plan_bytes = args.plan.read_bytes()
        plan_hash = __import__("hashlib").sha256(plan_bytes).hexdigest()
        plan = json.loads(plan_bytes)
        report = json.loads(args.report.read_text(encoding="utf-8"))
        verdict = evaluate(report, plan)
        if plan_hash != PLAN_SHA256:
            verdict = {**verdict, "status": "failed_incomplete", "shared_memory_gate": "failed_incomplete",
                       "failure_count": verdict["failure_count"] + 1,
                       "failures": [*verdict["failures"], {"path": "plan.sha256", "reason": "input plan bytes differ from the approved plan"}]}
        result = {**report, "verdict": verdict} if isinstance(report, dict) else {"report": report, "verdict": verdict}
    except (OSError, json.JSONDecodeError) as error:
        verdict = {"status": "failed_incomplete", "shared_memory_gate": "failed_incomplete",
                   "resident_vram_gate": "failed_preserved", "failure_count": 1,
                   "failures": [{"path": "input", "reason": f"cannot read valid plan/report JSON: {error}"}]}
        result = {"verdict": verdict}
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(serialized)
        except FileExistsError:
            parser.error(f"refusing to overwrite existing verdict: {args.output}")
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if verdict["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
