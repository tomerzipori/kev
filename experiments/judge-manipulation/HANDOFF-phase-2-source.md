# Phase 2 source checkpoint

This is a source and runtime milestone. The actual two-model, 200-row pilot has not run, and Phase 2 has not passed. No final suite, baseline, candidate training, or locked-test result is claimed.

`SOURCE-CHECKPOINT-phase-2.json` records the exact reviewed source, runtime, selection, index, and proof hashes. Its SHA-256 is `a4b2c38e2d248a65d3cc78a587d54a8a05e6d835d55aa6869e951dbfb4fcf34e`. Historical plan, registration, source audit, model lock, and Phase 4 evidence remain unchanged. The unrelated agent configuration file was not read or changed.

The retained candidate JSONL still has SHA-256 `bcb35908610e50c3e2cc1c787d1af1118735bcd67b9373af17ed595c1252939f`. It contains 160 programmatic rows and 40 label-unknown OASST calibration candidates. Its 43 lineage components comprise three synthetic components and 40 OASST conversation trees. States have 3 to 2,022 tokens including the delimiter. Candidate payloads and model outputs stay in ignored runs; the compact index, source selection, and reconstruction manifest are reviewable artifacts.

The Luna worker completed the narrow source repairs. Parent-tokenizer overrides verify the six historical file hashes. The pipeline rejects conflicting exact copies and cross-reservation lineage, validates strict boolean labels and exact evidence, checks complete chat context, and binds checker logs to candidate, response, runtime, prompt, and model hashes. Agreed OASST records pass the real materializer, strict encoder, and temporary Kev suite loader. The nonvacuous gate requires accepted known positives, known negatives, and OASST agreement.

The worker reported 24 focused tests passed in 96.48 seconds, then the added OASST loader regression passed separately in 97.15 seconds with 24 tests deselected. Compilation passed. A combined 25-test run and the full fast suite are not claimed. The original CPU prepare succeeded on October 16; the exact candidate hash was rechecked on October 18. Full prepare with the final portable override remains a runnable verification step.

Two original runtime failures are retained separately. A wrong cudart library directory caused CPU fallback, which the GPU guard rejected. After correcting the directory, CUDA inference aborted while the WSL GPU bridge was blocked. Root recovered that bridge with a controlled VM restart. One fresh request per checker then ran sequentially. Qwen completed on 37/37 GPU layers and passed JSON-schema, identity, and context assertions, but its report failed reading a live DrvFs log. Its raw response was not persisted; `qwen-recovery-proof-phase-2.json` states that gap. Phi passed the full harness on 33/33 layers with exact JSON and a persisted report. Both servers exited and the GPU was released to root. The gap in generic Qwen log retention does not supply any pilot label.

After the reviewed source/runtime checkpoint, reconstruct on a machine with the pinned parent tokenizer:

```sh
python experiments/judge-manipulation/phase2_pilot.py prepare --parent-tokenizer <pinned-tokenizer-directory>
```

The reconstruction must match the frozen candidate bytes. Start the actual checker passes only when root returns GPU ownership, persist every bound result, then derive dispositions and assess the registered gate. Use Linux-cache paths for active server logs. Phase 4 owns the next GPU window for the amended training probe. Phase 3 must audit effective attack/negative group counts and confidence-interval feasibility before freezing a corpus.
