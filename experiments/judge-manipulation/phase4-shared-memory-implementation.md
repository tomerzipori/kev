# Shared-memory benchmark implementation brief

Implement only the new plan in `phase4-shared-memory-plan.json`, after root reviews it. Work in this isolated worktree. Keep the primary checkout and its Linux `.venv` unchanged. Git operations belong to the Git specialist.

Most implementation belongs to one GPT-6-Luna xhigh worker once root releases its slot. Use uniquely named new setup, correctness, instrumentation and orchestration files. Reuse canonical Kev checkpoint, encoder, trainer, digest, device and kernel-environment helpers. Historical Phase 4 scripts and evidence are frozen. Read `unslop`, `kev-verify` and `test-audit` before relevant work; no test-only core hooks.

Acquire the four pinned wheels, verify their hashes and METADATA, and install with `--no-deps --target` into a fresh external overlay. Prepend it and this worktree to `PYTHONPATH` when using the original primary Python. Record imported paths, versions, torch ABI and the primary dependency hashes before and after. Leave Torch, Transformers, PEFT and Triton unchanged. The PyPI FLA distribution requires the separate `fla-core` wheel. Do not copy a Hopper-specific Triton override onto Ada.

Add the small separate GPU operator check described by the plan. It must exercise real backward gradients against the installed undecorated Transformers references. Keep Kev's inference fusion and CUDA graphs disabled. The full model may start only after this check passes, under the same 45-minute deadline.

Use a CPU-only parent watchdog with native Windows memory counters and WSL process sampling. Export allocator measurements from the measured child, without creating an extra CUDA context in the parent. The watchdog must enforce every registered threshold during correctness, training and reload. Preserve independent samples and a stop record if any query or stage fails. Verify process identity before signaling; never stop Phase 2 or unrelated applications.

Run the unchanged canonical trainer recipe for two updates, verify actual parameter changes, then use a fresh process for unmerged reload parity. Retain stage transitions and per-update timing. A pass needs all required evidence; two updates do not establish full-study reliability.

Before GPU execution, send the exact plan hash, source hashes, overlay inventory, CPU checks and proposed command to the director/root. Wait for root's explicit GPU handoff. Perform no broad refactor, frozen-file rewrite, package upgrade, driver change, paid computation or alternate model/precision/context attempt. Document a concrete failure rather than widening the plan after a result.
