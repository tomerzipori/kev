"""Exercise the serialized watchdog receipt consumed by the Phase 4 gate."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments/judge-manipulation"
GATE = EXPERIMENTS / "phase4_shared_memory_gate.py"
PLAN = EXPERIMENTS / "phase4-shared-memory-plan.json"


def complete_receipt():
    overlay = "/overlay"
    environment = {
        "gpu": "RTX 4060 Ti",
        "packages": {"torch": "2.8.0+cu128", "transformers": "5.17.0", "peft": "0.21.0",
                     "flash-linear-attention": "0.5.2", "triton": "3.4.0", "causal-conv1d": "1.7.0"},
        "deltanet": {"causal_conv1d_fn": "causal_conv1d.interface.causal_conv1d_fn",
                     "torch_chunk_gated_delta_rule": "fla.ops.gated_delta_rule.chunk_gated_delta_rule"},
    }
    model = {
        "base": "Qwen/Qwen3.5-4B-Base",
        "base_revision": "1001bb4d826a52d1f399e183466143f4da7b741b",
        "init_from": "jaredpalmer/kev-4b@139fdd94f1b6a6ad80cc15e08fcb99cac885a101",
        "lora_ranks": {"default": 16},
        "lora_targets_argument": "all",
        "option_isolation": False,
        "special_embeddings": False,
        "full_weight_training": False,
        "recipe_arguments": {"--head_dim": "256", "--batch": "1", "--accum": "16", "--checkpointing": "1",
                             "--epochs": "1",
                             "--lr": "2e-5", "--lora": "16", "--lora_targets": "all", "--dtype": "bf16",
                             "--weights_dtype": "bf16", "--max_state": "2048", "--seed": "0",
                             "--p_none": "0", "--p_none_distract": "0", "--p_distract": "0", "--p_none_pair": "0",
                             "--synthetic_repeat": "1", "--public_frac": "1", "--max_steps": "2"},
        "parameter_dtype_evidence": {
            "frozen_parameter_dtype_counts": {"bfloat16": 100, "float32": 3},
            "lora_trainable_parameter_dtype_counts": {"float32": 20},
        },
        "head_dtypes": ["float32"],
        "kernel_environment": environment,
        "kernel_bindings": {
            "causal_conv1d_fn": {"qualified_name": "causal_conv1d.interface.causal_conv1d_fn",
                                 "source": f"{overlay}/causal_conv1d/interface.py"},
            "torch_chunk_gated_delta_rule": {"qualified_name": "fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule",
                                              "source": f"{overlay}/fla/ops/gated_delta_rule/chunk.py"},
        },
        "overlay_path": overlay,
    }
    sample = {
        "cuda_active": True,
        "cuda_peak_allocated_bytes": 8 * 1024**3,
        "cuda_peak_reserved_bytes": 9 * 1024**3,
        "windows_adapter_shared_bytes": 2 * 1024**3,
        "active_model_process_rss_bytes": 4 * 1024**3,
        "windows_available_physical_bytes": 12 * 1024**3,
        "wsl_memavailable_bytes": 8 * 1024**3,
        "wsl_swap_used_bytes": 0,
        "gpu_dedicated_used_bytes": 3 * 1024**3,
        "gpu_dedicated_free_bytes": 5 * 1024**3,
    }
    return {
        "plan_sha256": "0329f3bc4fc6e9779caf1b32b393729b9f36fc6d28de85bb17e7c1b3bae4b48a",
        "model_lock_sha256": "0bdce0513f5d85c09154f6a8808ad973bc2848d679d1144c7093e64dc6d8fb19",
        "resident_vram_gate": "failed_preserved",
        "stop_reason": None,
        "execution": {"elapsed_seconds": 120, "driver_exit_code": 0, "terminal_stage": "complete"},
        "stages": {
            "kernel_correctness": {"status": "passed", "separate_process": True},
            "training": {"status": "passed", "fresh_process": True},
            "reload": {"status": "passed", "fresh_process": True},
        },
        "fixture": {
            "sha256": "ccc4c494c47e2559334b93842ce30c218da0e0fb6234eaf8d111aa2c5c97be20",
            "record_count": 32,
            "unique_source_records": 1,
            "state_tokens_including_state_delimiter": 2048,
            "full_encoded_row_tokens": 2099,
            "branch_tokens": 51,
        },
        "training": {
            "metrics": {
                "optimizer_steps": 2,
                "records_seen": 32,
                "step_seconds": [40.0, 25.0],
                "grad_norm": [{"steps": 2, "mean": 0.5, "max": 1.0}],
            },
            "timing": {"microbatches": 32, "optimizer_updates": 2,
                       "per_update_gradient_norms": [0.5, 0.8],
                       "optimizer_operation_seconds": [1.0, 1.1],
                       "optimizer_window_seconds": [20.0, 22.0]},
            "warm_start_coverage": {"tensors": 100},
            "model": model,
            "checkpoint_saved": True,
        },
        "kernel_correctness": {
            "status": "passed",
            "overlay_path": overlay,
            "package_versions": {"flash-linear-attention": "0.5.2", "fla-core": "0.5.2",
                                 "causal-conv1d": "1.7.0", "triton": "3.4.0"},
            "imported_paths": {"fla": f"{overlay}/fla/__init__.py",
                               "causal_conv1d": f"{overlay}/causal_conv1d/__init__.py"},
            "optimized_functions": {
                "delta_rule": {"module": "fla.ops.gated_delta_rule.chunk", "source": f"{overlay}/fla/ops/gated_delta_rule/chunk.py"},
                "causal_conv1d": {"module": "causal_conv1d.interface", "source": f"{overlay}/causal_conv1d/interface.py"},
            },
        },
        "reload": {
            "save_reload": {"exact_logits": True, "max_abs_logit_delta": 0},
            "temperature_pre_save": 1.0,
            "temperature_meta_pre_save": 1.0,
            "temperature_reload": 1.0,
            "eval_state_pre_save": {"model_training": False},
            "eval_state_reload": {"model_training": False},
            "tf32_pre_save": {"matmul": True, "cudnn": True},
            "tf32_reload": {"matmul": True, "cudnn": True},
            "kernel_environment_pre_save": environment,
            "kernel_environment_reload": environment,
            "unmerged_lora": True,
            "changed_adapter_tensor_count": 1,
            "changed_head_tensor_count": 1,
        },
        "monitoring": {
            "status": "complete",
            "max_gap_seconds": 1.0,
            "max_consecutive_missing_samples": 0,
            "missing_sample_count": 0,
            "max_consecutive_violating_samples": 0,
            "violating_sample_count": 0,
            "samples": [{**sample, "monotonic_ns": 1_000_000_000}, {**sample, "monotonic_ns": 2_000_000_000}],
        },
        "cleanup": {"reaped": True, "process_group_empty": True,
                    "windows_sampler": {"stopped": True, "reader_stopped": True, "exit_code": 0}},
        "source_hashes": {"runner": "a" * 64, "child": "b" * 64, "gate": "c" * 64, "plan": "d" * 64},
        "source_hashes_verified": True,
        "historical_files": {"before": {"report": "e" * 64}, "after": {"report": "e" * 64}, "unchanged": True},
    }


class SharedMemoryGateTests(unittest.TestCase):
    def invoke_gate(self, receipt, *, plan_bytes=None, preexisting_output=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        report_path, output_path = root / "report.json", root / "verdict.json"
        report_path.write_text(json.dumps(receipt), encoding="utf-8")
        plan_path = PLAN
        if plan_bytes is not None:
            plan_path = root / "plan.json"
            plan_path.write_bytes(plan_bytes)
        if preexisting_output is not None:
            output_path.write_text(preexisting_output, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(GATE), "--plan", str(plan_path), "--report", str(report_path), "--output", str(output_path)],
            capture_output=True, text=True, check=False,
        )
        stored = json.loads(output_path.read_text(encoding="utf-8")) if preexisting_output is None else None
        return result, stored["verdict"] if stored else None, output_path

    def test_complete_serialized_receipt_passes(self):
        result, verdict, _ = self.invoke_gate(complete_receipt())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(verdict["status"], "passed")
        self.assertEqual(verdict["resident_vram_gate"], "failed_preserved")

    def test_single_missing_allocator_sample_is_retained_under_three_sample_policy(self):
        receipt = complete_receipt()
        receipt["monitoring"]["samples"][0]["cuda_peak_allocated_bytes"] = None
        receipt["monitoring"]["max_consecutive_missing_samples"] = 1
        receipt["monitoring"]["missing_sample_count"] = 1
        result, verdict, _ = self.invoke_gate(receipt)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(verdict["status"], "passed")

    def test_three_consecutive_limit_violations_fail(self):
        receipt = complete_receipt()
        for index in range(3):
            receipt["monitoring"]["samples"].append({
                **receipt["monitoring"]["samples"][-1], "monotonic_ns": (index + 3) * 1_000_000_000,
                "windows_adapter_shared_bytes": 9 * 1024**3,
            })
        receipt["monitoring"]["max_consecutive_violating_samples"] = 3
        receipt["monitoring"]["violating_sample_count"] = 3
        result, verdict, _ = self.invoke_gate(receipt)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(verdict["status"], "failed_incomplete")

    def test_one_threshold_crossing_is_reported_without_changing_stop_policy(self):
        receipt = complete_receipt()
        receipt["monitoring"]["samples"][0]["windows_adapter_shared_bytes"] = 9 * 1024**3
        receipt["monitoring"]["max_consecutive_violating_samples"] = 1
        receipt["monitoring"]["violating_sample_count"] = 1
        result, verdict, _ = self.invoke_gate(receipt)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(verdict["resource_threshold_crossings_below_stop_streak"], 1)

    def test_synthetic_missed_poll_does_not_hide_actual_sample_gap(self):
        receipt = complete_receipt()
        first, second = receipt["monitoring"]["samples"]
        first["sample_kind"] = "resource_poll"
        second["sample_kind"] = "resource_poll"
        second["monotonic_ns"] = 3_000_000_000
        receipt["monitoring"]["samples"] = [
            first,
            {"sample_kind": "missed_poll", "monotonic_ns": 2_000_000_000, "cuda_active": False},
            second,
        ]
        receipt["monitoring"]["max_gap_seconds"] = 1.0
        receipt["monitoring"]["max_consecutive_missing_samples"] = 1
        receipt["monitoring"]["missing_sample_count"] = 1
        result, verdict, _ = self.invoke_gate(receipt)
        self.assertEqual(result.returncode, 2)
        self.assertIn("monitoring.max_gap_seconds", {failure["path"] for failure in verdict["failures"]})

    def test_unclean_windows_sampler_fails_the_receipt(self):
        receipt = complete_receipt()
        receipt["cleanup"]["windows_sampler"]["exit_code"] = 1
        result, verdict, _ = self.invoke_gate(receipt)
        self.assertEqual(result.returncode, 2)
        self.assertIn("cleanup.windows_sampler", {failure["path"] for failure in verdict["failures"]})

    def test_malformed_nested_evidence_fails_as_json_not_traceback(self):
        receipt = complete_receipt()
        receipt["training"]["metrics"]["step_seconds"] = "broken"
        receipt["training"]["timing"]["per_update_gradient_norms"] = [{"steps": "two"}]
        result, verdict, _ = self.invoke_gate(receipt)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(verdict["status"], "failed_incomplete")
        self.assertNotIn("Traceback", result.stderr)

    def test_output_path_refuses_overwrite(self):
        sentinel = '{"keep": true}\n'
        result, _, path = self.invoke_gate(complete_receipt(), preexisting_output=sentinel)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(path.read_text(encoding="utf-8"), sentinel)

    def test_malformed_plan_fails_closed(self):
        result, verdict, _ = self.invoke_gate(complete_receipt(), plan_bytes=b'{"limits": []}')
        self.assertEqual(result.returncode, 2)
        self.assertEqual(verdict["status"], "failed_incomplete")
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
