# Phase 4 feasibility gate

Status: **failed memory budget** (`failed_memory_budget`). The measured Windows/WSL desktop exceeded the registered physical 8 GiB gate. The trainer was stopped by a supervisor-directed SIGINT after the measured allocation crossed the limit; this was not a CUDA OOM and was not user cancellation.

The execution report at `experiments/judge-manipulation/report-phase-4.json` remains the immutable raw attempt outcome (`training_failure`, `KeyboardInterrupt`). Model loading and strict warm start succeeded. The terminal stage was forward at `2026-10-16T06:55:10.860087+00:00` UTC. Backward was observed earlier, but no completed optimizer update is claimed because no training metrics or step-completion evidence was produced. No checkpoint save or save/reload proof completed.

The allocator limit was 8,585,216,000 bytes (8 GiB capped by the CUDA device's reported 8,585,216,000 physical bytes). The observed allocated peak was 10,424,460,288 bytes, exceeding the limit by 1,839,244,288; reserved peak was 10,720,641,024 bytes. CUDA reported zero free bytes at the supervisor stop. NVIDIA sampling reached 7,909 MiB used with 41 MiB minimum free. Windows WDDM counters during the run showed VM-worker dedicated usage of 7,946,473,472 bytes and shared usage of 2,971,664,384 bytes. Those counters map to WSL's `vmwp` process PID 2188 and cannot be attributed exclusively to Linux trainer PID 10274.

The probe loaded the pinned Kev-4B parent into Qwen3.5-4B-Base with BF16 backbone/autocast, FP32 rank-16 LoRA and head, batch 1, accumulation 16, checkpointing, standard PyTorch, and a tokenizer-verified 2,048-token state. The fixture has 2099 encoded row tokens including the question branch and is one synthetic record repeated 32 times; it is hardware evidence only, not a corpus result.

The first fast suite ended with 1 failure, 475 passes, 26 skips, and 2 errors. Three affected nodes hit permissions under the root-owned default Hugging Face cache. Those three nodes passed on a focused rerun with the user-owned cache; the full suite was not rerun.

See `experiments/judge-manipulation/windows-gpu-counters-phase-4.json` for the timestamped portable Windows counter sample, `experiments/judge-manipulation/memory-gate-stop-phase-4.json` for the stop record, and `experiments/judge-manipulation/HANDOFF-phase-4.md` for reproducibility, validation, artifact hashes, and the next gate. No corpus build, baseline evaluation, or locked test was run. Phase 2 and Phase 3 remain pending.
