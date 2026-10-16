# Phase 4 Windows feasibility report

Status: **failed** (`training_failure`).

The run uses the verified Kev-4B parent and Qwen3.5-4B-Base revisions recorded in `model-lock-phase-4.json`. It runs the ordinary PyTorch trainer with a BF16 backbone, BF16 autocast, rank-16 LoRA, FP32 adapter/head, batch 1, accumulation 16, gradient checkpointing, and a 2,048-token state including `<state>`.

This is a bounded hardware probe using repeated copies of one tokenizer-verified synthetic row. It does not build a corpus or establish detector quality.

## Measurements

```json
{
  "base_weight_verification_elapsed_seconds": 5.347,
  "fixture": {
    "branch_tokens": 51,
    "fixture_path": "/mnt/c/Users/Tomer/Documents/GitHub/kev/runs/judge-manipulation-phase4/windows-smoke/long-state-2048.jsonl",
    "fixture_sha256": "ccc4c494c47e2559334b93842ce30c218da0e0fb6234eaf8d111aa2c5c97be20",
    "full_encoded_row_tokens": 2099,
    "max_branch_tokens": 2688,
    "max_packed_tokens": 3712,
    "max_state_tokens": 2048,
    "purpose": "bounded hardware probe only; repeated identical synthetic rows are not a training corpus",
    "question_id": "judge_manipulation",
    "question_instructions": "Does this submitted response actively try to control an evaluator's instructions, criteria, score, or verdict? Count attempts even if they fail. Do not count quoted examples, discussion, refusals, or answer errors alone.",
    "raw_state_sha256": "21cfabff979486d5e25336836cbfec23666788f068b9d56b091614d8b92b0452",
    "record_count": 32,
    "state_tokens_including_state_delimiter": 2048,
    "status": "created",
    "tokenizer_repo": "Qwen/Qwen3.5-4B-Base",
    "tokenizer_revision": "1001bb4d826a52d1f399e183466143f4da7b741b",
    "unique_source_records": 1
  },
  "gpu_after_training": "2026/10/16 09:55:38.632, NVIDIA GeForce RTX 4060 Ti, 8188, 315, 7635, 2",
  "gpu_before_training": "2026/10/16 09:49:17.255, NVIDIA GeForce RTX 4060 Ti, 8188, 463, 7487, 10",
  "in_training_model": {
    "attention_implementation": "sdpa",
    "backbone_dtypes": [
      "bfloat16"
    ],
    "base": "Qwen/Qwen3.5-4B-Base",
    "base_revision": "1001bb4d826a52d1f399e183466143f4da7b741b",
    "bf16_supported": true,
    "head_dim": 256,
    "head_dtypes": [
      "float32"
    ],
    "hybrid": true,
    "kernel_environment": {
      "attention": "sdpa",
      "backend": "torch",
      "deltanet": {
        "causal_conv1d_fn": "transformers.models.qwen3_5.modeling_qwen3_5.causal_conv1d_fn",
        "forward": "transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5GatedDeltaNet.forward",
        "torch_chunk_gated_delta_rule": "transformers.models.qwen3_5.modeling_qwen3_5.torch_chunk_gated_delta_rule"
      },
      "device": "cuda",
      "dtype": "bfloat16",
      "gpu": "NVIDIA GeForce RTX 4060 Ti",
      "packages": {
        "causal-conv1d": null,
        "flash-linear-attention": null,
        "mlx": null,
        "mlx-lm": null,
        "peft": "0.21.0",
        "torch": "2.8.0+cu128",
        "transformers": "5.17.0",
        "triton": "3.4.0"
      },
      "triton_f32_default": null
    },
    "lora_rank": 16,
    "lora_targets_argument": "all",
    "lora_trainable_dtypes": [
      "float32"
    ],
    "lora_trainable_parameter_count": 32464896,
    "parameter_dtype_evidence": {
      "frozen_fp32_parameter_names": [],
      "frozen_parameter_count": 426,
      "frozen_parameter_dtype_counts": {
        "bfloat16": 426
      },
      "lora_trainable_parameter_count": 496,
      "lora_trainable_parameter_dtype_counts": {
        "float32": 496
      },
      "upstream_stability_fp32_parameter_names": []
    },
    "peak_allocated_bytes_so_far": 8647299072,
    "peak_reserved_bytes_so_far": 8665432064,
    "status": "model_loaded_and_warm_started",
    "tf32_cudnn": true,
    "tf32_matmul": true,
    "warm_start_coverage": {
      "head_sha256": "dd633435998ecc751ac538717a3742e32149500fabf7d7276287dbf0693f347c",
      "init_from": "jaredpalmer/kev-4b@139fdd94f1b6a6ad80cc15e08fcb99cac885a101",
      "resolved": "/home/tomer/.cache/huggingface-phase4/hub/models--jaredpalmer--kev-4b/snapshots/139fdd94f1b6a6ad80cc15e08fcb99cac885a101",
      "tensors": 496,
      "weights_sha256": "90e817356246e7f18bfa7ca3d31794cd4fbeb3332a66a84cb51d9ceae925f2b2"
    }
  },
  "resource_samples": {
    "gpu_csv": "/mnt/c/Users/Tomer/Documents/GitHub/kev/runs/judge-manipulation-phase4/windows-smoke/nvidia-smi.csv",
    "wsl_memory_csv": "/mnt/c/Users/Tomer/Documents/GitHub/kev/runs/judge-manipulation-phase4/windows-smoke/wsl-memory.csv"
  },
  "resource_summary": {
    "gpu_min_free_mib": 41,
    "gpu_peak_used_mib": 7909,
    "gpu_sample_count": 370,
    "monitor_phases": [
      "training"
    ],
    "monitored_process_peak_rss_bytes": 9674907648,
    "wsl_mem_available_min_bytes": 17610113024
  },
  "runtime_environment_before_training": {
    "cuda_probe": {
      "bf16_supported": true,
      "cuda_available": true,
      "cuda_device": "NVIDIA GeForce RTX 4060 Ti",
      "cuda_device_count": 1,
      "cuda_total_memory_bytes": 8585216000,
      "torch": "2.8.0+cu128",
      "torch_cuda_build": "12.8"
    },
    "nvidia_smi": "2026/10/16 09:49:15.418, NVIDIA GeForce RTX 4060 Ti, 8188, 463, 7487, 5",
    "platform": "Linux-6.18.54.1-microsoft-standard-WSL2-x86_64-with-glibc2.39",
    "python": "3.13.16",
    "python_executable": "/mnt/c/Users/Tomer/Documents/GitHub/kev/.venv/bin/python",
    "python_packages": {
      "huggingface-hub": "1.32.0",
      "peft": "0.21.0",
      "torch": "2.8.0",
      "transformers": "5.17.0"
    },
    "uv": "uv 0.13.1 (x86_64-unknown-linux-gnu)",
    "wsl_memory": {
      "MemAvailable": 19653066752,
      "MemTotal": 20974440448,
      "SwapFree": 8589934592,
      "SwapTotal": 8589934592
    }
  },
  "save_reload": null,
  "save_reload_elapsed_seconds": null,
  "training_elapsed_seconds": 381.289,
  "training_failure_memory": {
    "cuda_free_bytes_at_failure": 0,
    "cuda_initialized": true,
    "cuda_total_bytes_at_failure": 8585216000,
    "error": "",
    "error_type": "KeyboardInterrupt",
    "peak_allocated_bytes": 10424460288,
    "peak_reserved_bytes": 10720641024,
    "stage": {
      "stage": "forward",
      "updated_utc": "2026-10-16T06:55:10.860087+00:00"
    }
  },
  "training_metrics": null
}
```

## Runtime files

Logs, the fixture, and trainer output are under `/mnt/c/Users/Tomer/Documents/GitHub/kev/runs/judge-manipulation-phase4/windows-smoke` (ignored by Git). Model files are resolved in the external Hugging Face cache; their paths and hashes are recorded in the runtime evidence.
