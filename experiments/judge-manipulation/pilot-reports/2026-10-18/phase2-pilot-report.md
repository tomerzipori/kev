# Phase 2 pilot report — 2026-10-18

The registered **structural nonvacuity gate passed**: the frozen 200 rows have one disposition each, both pinned local checkers produced 200 bound records, and the run accepted known positives, known negatives, and OASST semantic agreements with validated evidence. **This does not establish corpus-scale checker quality, group generalization, Phase 3 readiness, or training readiness.** Only 9 of 200 rows were accepted.

## Run results

| Checker | Bound rows | Valid supported | Parser-rejected | Valid `supported=false` | Truncated |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B Q4_K_M | 200 | 175 | 17 | 1 | 7 |
| Phi-4-mini Q4_K_M | 200 | 35 | 93 | 70 | 2 |

Both passes had zero transport/inference exceptions. The truncated counts above are `finish_reason=length`; they were not retried. Every ledger row passed the frozen candidate/runtime/prompt/model/response hash binding during disposition building, and each checker had one consistent live runtime identity.

| Disposition | Rows |
|---|---:|
| Accepted known construction | 5 (3 positive, 2 negative) |
| Accepted OASST semantic agreement | 4 |
| Checker disagreement | 22 |
| Malformed or unsupported | 162 |
| Inference failure | 7 |

The four OASST agreements all received a negative label. Their materialized Kev records passed the real loader and encoder; the file and validated partition SHA-256 is `0234461eb6d09d7a73ebd83ba69f01476911377633f825e715d925961d0b0c71`.

### Rejection reasons

The 162 malformed/unsupported dispositions were first rejected for these reasons:

| First rejection | Rows |
|---|---:|
| `one or more checkers marked the response unsupported` | 66 |
| `phi4-mini-q4km: checker evidence must be a non-empty array` | 47 |
| `phi4-mini-q4km: evidence string is not present verbatim in the original response` | 20 |
| `qwen3-4b-q4km: evidence string is not present verbatim in the original response` | 16 |
| `phi4-mini-q4km: unsupported checker output must set manipulation=false` | 11 |
| `phi4-mini-q4km: checker output must have exactly supported, manipulation, and evidence fields` | 1 |
| `qwen3-4b-q4km: unsupported checker output must set manipulation=false` | 1 |

Per-checker totals show the main failure modes. Qwen had 16 nonverbatim evidence spans and one invalid `supported=false`/`manipulation=true` combination. Phi had 60 empty evidence arrays, 21 nonverbatim spans, 11 invalid unsupported/positive combinations, and one extra-field response; 70 additional schema-valid Phi outputs said `supported=false`. These are outputs that fail the frozen strict response contract; no clear parser defect was found. The prompt and parser were unchanged.

## Coverage and limits

- The 160 constructed rows contain 80 attacks and 80 controls across eight classes (20 rows each), with 24 reservation-specific template families and 80 answer bodies.
- Five placements have 32 rows each; there are 16 attacks after token 512 (positions 513–532) and 16 near the 2,048-token cap (positions 1,901–1,919). State lengths range from 43 to 2,022 tokens.
- Six control types contribute 26–28 rows each. Seventeen of 20 obfuscation attacks include format/invisible Unicode; seven rows contain literal role markers.
- There are 43 effective lineage groups: 3 synthetic components and 40 OASST conversation trees. The synthetic rows collapse to one group per reservation, so this pilot does not provide useful independent group-level evidence for synthetic patterns.

The large malformed/unsupported quarantine rate and low synthetic effective-group count leave checker usability and source-level generalization unresolved. No accuracy, performance, training, or final Phase 2 quality claim is made. Do not proceed to Phase 3 freeze or training until those limits are reviewed. The root may continue the separate Phase 4 source checkpoint and coordinate its GPU benchmark after the source preflight.

## Runtime and retained artifacts

The two local WSL checkers ran sequentially on the RTX 4060 Ti using llama.cpp build 11425. Final Linux-cache server logs show Qwen offloaded **37/37 layers** and Phi offloaded **33/33 layers**; their hashes and all frozen input/output hashes are recorded in [the machine-readable report](phase2-pilot-report.json). Both servers were stopped after their single pass, localhost port 18082 was free, and GPU memory returned to the desktop compositor. Raw response/candidate ledgers and server logs remain outside Git; this report contains aggregate counts only.

The report also binds the candidate file `bcb35908610e50c3e2cc1c787d1af1118735bcd67b9373af17ed595c1252939f`, Qwen ledger `d7dd33ba85723d2f49af7a0760802cc2c921273d0c8b34e06ec6cfa220207a25`, Phi ledger `79e1c4a8836050f9a404d8b408d5aa13d2d2ae698f7a4c82e68dc14a77c7456e`, and dispositions `c3b924e4e7d6e9b8da54af40f6ba8817776e7d28524d4895a0ad98fc479c9f0d` by SHA-256. No prompt, label, runtime, or retry changes were made.
