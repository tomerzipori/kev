# Phase 4 handoff

The Phase 4 Windows feasibility gate **failed on the measured desktop**. Allocated memory peaked at 10,424,460,288 bytes against a limit of 8,585,216,000 bytes; the measured excess was 1,839,244,288 bytes. Reserved memory peaked at 10,720,641,024 bytes. The exact run is recorded in the immutable raw report, and the separate gate record preserves the memory-budget decision. No Phase 2 or Phase 3 transition follows from this run.

The model lock is Kev-4B parent `jaredpalmer/kev-4b@139fdd94f1b6a6ad80cc15e08fcb99cac885a101`, on `Qwen/Qwen3.5-4B-Base@1001bb4d826a52d1f399e183466143f4da7b741b`. User explicitly confirmed the 4B objective. The lock SHA-256 is `0bdce0513f5d85c09154f6a8808ad973bc2848d679d1144c7093e64dc6d8fb19` and was unchanged before and after base shard verification. Actual downloaded base shards matched the lock: `model.safetensors-00001-of-00002.safetensors` `df547074dce70532a0493e5433152bd17a65efb89088cfabc2e7e2371a93d712` and `model.safetensors-00002-of-00002.safetensors` `590fbaac095dd31db886c322d9d2f7df47777966391acf306ddddc3e4e3a15ef`. Adapter, head, and tokenizer hashes are preserved in the model lock.

The probe used standard PyTorch with the registered LoRA feasibility recipe: BF16 frozen backbone and autocast, FP32 rank-16 LoRA adapter and pointer head, batch 1, accumulation 16, gradient checkpointing, 2,048 state tokens including `<state>`, and 32 repeats of one exact synthetic fixture. The whole encoded row was 2099 tokens including the 51-token question branch. No augmentation, QLoRA, context reduction, offload, optimized serving kernel, or model-size substitution was used. This fixture is not a corpus.

## What completed

- Dependency setup and public weight verification passed. CUDA and BF16 were available on the RTX 4060 Ti.
- Kev-4B loaded and strict warm start covered 496 adapter tensors and the pointer head. Observed frozen backbone parameters were BF16; LoRA and head parameters were FP32. The actual runtime used SDPA and the Transformers reference PyTorch DeltaNet implementation.
- The process reached the forward stage and had earlier entered backward. The root-directed graceful SIGINT stopped only trainer PID 10274 after memory evidence crossed the gate. The terminal exception is `KeyboardInterrupt`; CUDA free memory was zero and allocator peak was 10,424,460,288 bytes. This was not reported as an OOM.

## What remains unproven

No optimizer-step completion is asserted: the run did not write `training_metrics.json`, and the final stage was forward. No completed checkpoint-save or save/reload comparison was recorded. There is no corpus-quality result. The current desktop does not pass the registered resident-memory gate; a successful process exit on another machine must still be accompanied by allocator and physical-device evidence within the limit.

Windows `GPU Process Memory` counters observed `vmwp` PID 2188 at 2026-10-16T06:53:09.6870517Z: dedicated 7,946,473,472 bytes and shared 2,971,664,384 bytes. Adapter shared usage was 3,008,876,544 bytes. These are VM-level WDDM counters; no direct exclusive mapping from Linux PID 10274 to Windows PID 2188 is available. See `windows-gpu-counters-phase-4.json`.

## Validation

The initial locked fast suite completed with **1 failed, 475 passed, 26 skipped, 1 warning, and 2 errors**. The three affected nodes hit Hugging Face cache permission errors under `/home/tomer/.cache/huggingface`; the full suite was not rerun. The three nodes passed in a focused run (33.05 seconds) with the user-owned cache at `/home/tomer/.cache/huggingface-phase4`. The full suite is not reported as green. No tests were added or changed. The executed command was:

Executed focused command (with the four cache variables set to the user-owned cache):

```sh
uv run --project /mnt/c/Users/Tomer/Documents/GitHub/kev --locked --extra serve --python 3.13.16 python -m pytest tests/test_unit.py::test_user_text_cannot_forge_delimiters tests/test_unit.py::test_encode_positions_restart_per_branch tests/test_research.py::test_training_plan_matches_registered_screen -q
```

All 12 frozen file hashes in `BRANCH-phase-4.json` were rechecked with no mismatch. Python compilation passed for all five Phase 4 Python scripts, and `bash -n` passed for the WSL setup script. Report-only replay passed using the raw report while pointing at a nonexistent runtime directory. A read-only process check at 2026-10-16T07:08:18.9005126Z found zero Phase 4 trainer matches and no `nvidia-smi` CUDA compute applications; no process was stopped during this check.

## Replay the retained gate without runtime files

This command reads only the committed raw report and recalculates its memory gate. It points to a nonexistent runtime directory, uses no GPU, and needs no ignored proof files:

```sh
.venv/bin/python - <<'PY'
import importlib.util, json, sys
from pathlib import Path
root = Path.cwd()
sys.path.insert(0, str(root))
spec = importlib.util.spec_from_file_location("phase4_runner", root / "experiments/judge-manipulation/run_phase4_smoke.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
report = json.loads((root / "experiments/judge-manipulation/report-phase-4.json").read_text())
result = runner.memory_budget_assessment(report, Path("/tmp/phase4-runtime-not-present"))
assert result["status"] == "failed_memory_budget"
assert result["peak_allocated_bytes"] == 10424460288
print(json.dumps(result, sort_keys=True))
PY
```

## Reproduce on a suitable host

The WSL setup script is repository-relative, pins Python 3.13.16 and uv 0.13.1, uses a configurable user-owned HF cache, and runs `uv sync --locked`. The next step is root/user review of this failed gate and a CPU-only replay of the retained report. Do not advance Phase 2 or Phase 3 now. The exact 2,048-state-token fixture and its 51-token question branch cover the registered Phase 4 input shape. If a later Phase 4 attempt is approved, run from the repository root:

```sh
bash experiments/judge-manipulation/setup_phase4_wsl.sh
RUN_ID="phase4-$(date -u +%Y%m%dT%H%M%SZ)"
.venv/bin/python experiments/judge-manipulation/run_phase4_smoke.py \
  --out "runs/judge-manipulation-phase4/$RUN_ID" \
  --report-dir "experiments/judge-manipulation/reports/$RUN_ID"
```

Use a fresh `RUN_ID` on each attempt. The runner refuses to overwrite reports or run directories. The reported allocator peak must be at most `min(8 * 1024^3, actual CUDA physical device bytes)` and the full two-step/save-reload proof must complete before Phase 4 can pass. Do not proceed to Phase 2 or Phase 3 based on this desktop result.

## Artifact hashes

The runner version used for the raw attempt was source checkpoint `6ca8955`; it is hashed separately from the current post-run runner with the memory-gate fix. The handoff itself is not self-hashed.

- `experiments/judge-manipulation/run_phase4_smoke.py` — SHA-256 `31a75bc629fa1e0c7830e5bf198390bc3ad96107d3fa2f07cc331b3a1ace1c45`
- `experiments/judge-manipulation/run_phase4_smoke.py` — SHA-256 `da935cf6a058ea310461d4c2f97c9ce3b5fad36b3c9da4a52bd2a1199198a0c2` (source revision `6ca8955`)
- `runs/judge-manipulation-phase4/windows-smoke/long-state-2048.jsonl` — SHA-256 `ccc4c494c47e2559334b93842ce30c218da0e0fb6234eaf8d111aa2c5c97be20`
- `runs/judge-manipulation-phase4/windows-smoke/fixture.json` — SHA-256 `114441351e4906027223ecc0935660538bf4ecf4ddd77d08d850f24a036cb732`
- `runs/judge-manipulation-phase4/windows-smoke/focused-fast-cache-tests-correct-env.log` — SHA-256 `05a655ad92f64bf6d037a37c192087959af5dd2182e2d373a218a0ce454aa3ee`
- `runs/judge-manipulation-phase4/validation-fast-tests.log` — SHA-256 `81a597f5b800ce8c08904398279b7309b0b8e6a0ce0dd4accbac4641fea1044a`
- `runs/judge-manipulation-phase4/windows-smoke/proof/in_training_model.json` — SHA-256 `f86ddc9033612deae1dad85750f600ebcc6ef8ede2f1705d7b06ce706d3d196e`
- `experiments/judge-manipulation/phase4_train_instrumented.py` — SHA-256 `6bedf53a6dfa1f0e1595f73e80b527e8e5a78727d2d67be4cdff5591792a621a`
- `experiments/judge-manipulation/memory-gate-stop-phase-4.json` — SHA-256 `35ade5fc6aa681a93e689c758525d2db65012f818dfd37132a7b51c058b0bfef`
- `experiments/judge-manipulation/model-lock-phase-4.json` — SHA-256 `0bdce0513f5d85c09154f6a8808ad973bc2848d679d1144c7093e64dc6d8fb19`
- `runs/judge-manipulation-phase4/windows-smoke/postrun-idle-check.txt` — SHA-256 `12ff35b6a8e7d062fe377e65bc5931ac2f480f11eddbe53fb12f872f305f459b`
- `experiments/judge-manipulation/report-phase-4.json` — SHA-256 `cfe2571a4e8fd9bfdf35b1b0d649b50b20e5461e0f718ef1320c3a607cf5d49e`
- `experiments/judge-manipulation/report-phase-4.md` — SHA-256 `5297cfdae714a11c008dd57aa2cce9ae5db0595f659fe3b61dc9bd5b44c345c0`
- `experiments/judge-manipulation/verify_phase4_reload.py` — SHA-256 `50f8a31685261a3fb57af763b5ec3cd916269df99f8ae93b9b043ae35b7f40fa`
- `runs/judge-manipulation-phase4/windows-smoke/proof/training-failure-memory.json` — SHA-256 `c081fbbdd5bd5d567e4e0eececff1e1e77b78653aa812fcc231bd7c13ac03dc0`
- `experiments/judge-manipulation/windows-gpu-counters-phase-4.json` — SHA-256 `d92e08b5df78792f123edcf148c609528a6a13265a133e40a5d36e43529f5599`
- `runs/judge-manipulation-phase4/windows-smoke/windows-gpu-memory-summary.json` — SHA-256 `22d5312588a5611a259671fe9b1a2161edf2f9663bf19bbd1bb9c92c8fc3f16b`
- `experiments/judge-manipulation/feasibility-gate-phase-4.json` — SHA-256 `94028e5f6dc57593e343665c628b81d59c89d9f7bb25cd4cf59d091ac5adf206`
- `experiments/judge-manipulation/feasibility-gate-phase-4.md` — SHA-256 `de9aa06c14de714fd01188ed4b08b84e774b69c5e4249c9b12827fcfd595ce68`
Frozen plan SHA-256: `812f13d43f41e05597b09fd783a7486aa7999f80fd995db94815b826c17892ba`. Original raw report SHA-256: `cfe2571a4e8fd9bfdf35b1b0d649b50b20e5461e0f718ef1320c3a607cf5d49e`. No git mutation was performed by this worker.
