# Phase 2 checker runtime

The local runtime is a pinned llama.cpp CUDA 12.8 release build with one checker loaded at a time. The runtime and model pins, server protocol, hashes, and smoke status are in [`checker-runtime-lock-phase-2.json`](checker-runtime-lock-phase-2.json). This document records how to reproduce the setup and the evidence collected for this workstation.

## Setup

Run these commands from the repository root in WSL Ubuntu 24.04. Downloads, the llama.cpp checkout, the isolated conversion environment, and all model files stay under `/home/tomer/.cache`; the repository `.venv` is not changed.

```sh
bash experiments/judge-manipulation/setup_checker_runtime_phase2.sh runtime
bash experiments/judge-manipulation/setup_checker_runtime_phase2.sh models
bash experiments/judge-manipulation/setup_checker_runtime_phase2.sh converter-env
bash experiments/judge-manipulation/setup_checker_runtime_phase2.sh convert-phi
bash experiments/judge-manipulation/setup_checker_runtime_phase2.sh verify
```

`runtime` verifies both official `b11425` CUDA 12.8 release asset digests, extracts the binary and cudart bundle into the external cache, extracts the pinned Ubuntu `libgomp1` package without installing a system package, checks out the exact source commit used for conversion, and prints `llama-server --version`. Incomplete asset downloads use `.part` paths and only replace final archive paths after their expected digest matches.

`models` downloads only the pinned official Qwen Q4_K_M asset, the original Qwen tokenizer snapshot, and the original Microsoft Phi safetensors/tokenizer snapshot. `converter-env` syncs the hashed requirements lock into a separate Python 3.13.16 environment using uv 0.13.1. `convert-phi` verifies both Phi source shard hashes, converts those source weights locally to F16 GGUF with the pinned converter, then quantizes that result to Q4_K_M with the pinned `llama-quantize`. Partial GGUF outputs have attempt-specific names; only a successful command is renamed to the final model path. The downloaded source weights and finalized GGUFs remain outside Git.

The Phi conversion completed locally. The pinned converter produced a 7,680,694,592-byte F16 GGUF with SHA-256 `8c61a6508269df3163fbc7baf2771567f42abfd0727cb113aeefaab5e52b03d2`. The b11425 `llama-quantize` command then produced the 2,493,840,704-byte Q4_K_M GGUF with SHA-256 `6ec1dc61cdda56cd97a4cc22f3aee71569fa60c989368616be92e8ad223fa6ab`; its measured quantization time was 38.521 seconds. Both paths and digests are pinned in the lock and checked by the setup script before verification or serving.

The Hugging Face variables are set to the phase-specific user-owned cache before the downloader or converter imports run:

```text
HF_HOME=/home/tomer/.cache/huggingface-phase2
HF_HUB_CACHE=/home/tomer/.cache/huggingface-phase2/hub
HF_DATASETS_CACHE=/home/tomer/.cache/huggingface-phase2/datasets
```

The pinned PyPI sentencepiece 0.2.1 release publishes a CPython 3.13 manylinux x86_64 wheel, so this conversion environment does not require the absent system compiler.

## Serving and input-size preflight

Start only one model at a time. The server script binds loopback port 18082, disables its web UI, advertises the fixed alias expected by the Phase 2 client, uses an 8192-token context with one slot, requests all GPU layers, and enables flash attention and Jinja templates. Qwen thinking is disabled with `enable_thinking: false` in the server defaults and per-request template arguments.

```sh
bash experiments/judge-manipulation/setup_checker_runtime_phase2.sh serve-qwen
# Or stop the first process, confirm GPU memory is released, then:
bash experiments/judge-manipulation/setup_checker_runtime_phase2.sh serve-phi
```

The OpenAI-compatible chat base URL is `http://127.0.0.1:18082/v1`; the model IDs are `qwen3-4b-q4km` and `phi4-mini-q4km`. For preflight, send the complete system/user message array to `POST http://127.0.0.1:18082/apply-template` with `add_generation_prompt: true` and the exact same `response_format` JSON schema used for completion. Qwen requests also include `chat_template_kwargs: {"enable_thinking": false}`. The server returns the exact serialized prompt in `{"prompt": "..."}`. Send that prompt to `POST /tokenize` with `add_special: true`, `parse_special: true`, and `with_pieces: false`; the server returns `{"tokens": [...]}`. The pipeline must reserve completion `max_tokens` within the same 8192-token context and reject an oversized request before calling chat completion.

## Evidence

The official Qwen Q4_K_M asset was downloaded and its bytes match the pinned Hugging Face LFS SHA-256. The original Phi source weights were downloaded at their immutable Hub revision; both shard hashes match the pinned Hub LFS SHA-256 values. The Qwen model card declares Apache-2.0; the Microsoft Phi repo contains its MIT license file, pinned and hashed in the runtime lock.

The CUDA 12.8 llama.cpp release archive, cudart archive, and external `libgomp1` runtime package are pinned in the lock. The server binary reports release build 11425 at commit `e117148a4`; the source checkout uses the same commit. The standalone Phase 2 model smoke results, observed effective server arguments, loaded GPU layer counts, and rendered-template hashes are added here after execution. No registered pilot prompts are run by this setup step.

## Reliability notes

The first isolated converter-environment invocation installed all 42 locked packages but returned a transient shell `c: command not found` after printing the completed package freeze. An immediate idempotent rerun completed successfully, recorded the same 42-package environment, and passed the converter `--help` check. The first Phi conversion invocation successfully finalized the F16 intermediate and then stopped because the draft script assumed the release archive was flat; inspection showed the official archive’s executables live under `llama-b11425/`. The script now uses that verified nested path for version, quantization, serving, and shared libraries. The quantizer rerun consumed the preserved F16 intermediate and succeeded. No incomplete model file was mistaken for a finalized output.

## Current status

The converter environment and local Phi conversion/quantization are complete. On 2026-10-18 the first generic Qwen smoke reached valid JSON, loaded-model identity, and 8192-token context checks, then failed the GPU-offload guard. The cudart archive also contains a nested release directory; the script had pointed one directory too high and silently used CPU. The script now points to the actual cudart library directory.

With that correction, Qwen reached the healthy API through the CUDA backend, but its first generic completion aborted in `ggml_backend_cuda_synchronize` with `CUDA error: unknown error`. The server exited. WSL `nvidia-smi` then reported `GPU access blocked by the operating system`; native Windows `nvidia-smi` still reported the GPU. This is retained failure evidence, not a completed GPU smoke. Phi has not been loaded. Root owns GPU bridge recovery and any next training probe. No full pilot checker calls occurred.

Root then recovered the WSL GPU bridge with a controlled VM restart. One new generic request per checker ran sequentially. Qwen completed on 37/37 GPU layers at 73.69 generated tokens/s and passed the harness's JSON, identity, and context checks. The harness then failed to read its active Windows-mounted log with `OSError 61`. Its raw completion body was not persisted; `qwen-recovery-proof-phase-2.json` records that gap without inventing a complete success report. No Qwen request was repeated.

Phi used a Linux-cache runtime log and passed the full generic harness with 33/33 layers on GPU, the pinned model identity, 8192-token context, exact JSON `{"ok": true}`, and `finish_reason=stop`. Its completed report and log were copied after the process exited. Both servers stopped, and the GPU was released to root for the training probe.

The separate attempt logs and their hashes are recorded in `runtime-attempts-phase-2.json`. Root review and a source/runtime checkpoint still precede the unexecuted 200-item pilot. The generic smoke uses JSON Schema; the pilot must use the exact unchanged response format in `checker-prompts-phase-2.json`, currently `json_object`, in both preflight and completion. Active runtime logs should stay in the Linux cache to avoid concurrent DrvFs reads.
