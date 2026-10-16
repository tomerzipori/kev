# Phase 0 handoff

Phase 0 registration passes. The current `MASTER-PLAN.md` objective fixes the intended detector scope at Kev-4B and 32 GB RAM. The detailed recipe and released-model baseline still name Kev-0.8B. This handoff records Kev-4B as the intended scope, leaves the exact parent and baseline lock pending, and preserves the plan byte for byte.

Phase 1 source research may start because it does not depend on a model lock. Do not run a model baseline or training until the released Kev-4B checkpoint revision, Qwen base revision, adapter, pointer head, and target modules are verified. The plan's `bf75...` value is an inspected parent-checkpoint revision, not a Qwen base revision. Do not claim WSL2 feasibility until the training smoke runs on the RTX 4060 Ti through WSL2 Ubuntu.

## Completed registration

- [registration.json](registration.json) records the fixed objective, exact detector question, record and token contract, source rules, pilot and synthetic coverage, partition targets, four seed-0 trials, confirmation seeds, selection, acceptance, and reproducible statistical conventions.
- [README.md](README.md) explains the experiment in reviewer-facing prose and maps the plan's Mac paths to this Windows checkout.
- Preserved [git-state.json](git-state.json) and [git-workflow.md](git-workflow.md), created by the git-master audit. They record the original `main` state, pre-existing `MASTER-PLAN.md` edit, and local branch workflow. This registration adds the plan's current Kev-4B objective interpretation without rewriting that audit record.

The checkout is on `codex/judge-manipulation-phase-0-register`, created from `925e9e900ed7c04e40d38549eac32a0d974c57d8`. No commit or push was made. The pre-existing `MASTER-PLAN.md` edit remains the only tracked change.

## Implementation evidence

The Kev path can represent this task once the checkpoint is locked. `kev/api.py` accepts `noul` questions and maps them to false/true options. `kev/data.py` loads labeled request records and materializes them through that same API schema. `kev/train.py` has `--init_from`, LoRA, dtype, checkpointing, seed, learning-rate, and max-state options; warm start loads the parent adapter and pointer head. `kev/model.py` strict encoding rejects overflow and counts the `<state>` delimiter in the limit. Its `user_tokens` path rewrites caller-supplied delimiter-shaped strings before tokenization so response text cannot create Kev control tokens.

These were read-only source inspections. No Kev core files changed. The 2,048-token state limit is within the trainer's declared range. This inspection does not establish that the 4B model fits the target GPU.

The seed-0 grid has four trials: learning rates `2e-5` and `5e-5`, each at one and two epochs. They will share one verified parent after the model lock passes. Train, calibration, and development targets are each balanced 50/50. The best qualifying seed-0 checkpoint is selected by the registered development ranking; seeds 1 and 2 report stability only and cannot change the checkpoint, recipe, threshold, or temperature. Parent and candidate reads must use the same inference precision, kernel configuration, and device.

## Pending gates

- **Model lock:** replace the stale 0.8B recipe and baseline references with verified Kev-4B artifacts and exact revisions. Do not infer an artifact from a repository name.
- **Windows feasibility:** the supervising audit reported an RTX 4060 Ti with 8,188 MiB and driver 617.14, while WSL Ubuntu 24.04 was stopped. This is host metadata only, not a CUDA, BF16, dependency, checkpoint-load, or training result.
- **Phase 1:** audit each source's actual files, label meaning, lineage, and reuse terms; log admission and exclusions. The 200-row pilot remains a required gate before full corpus processing.
- **Phase 3:** freeze the suites only after source revisions and hashes, schemas, group IDs, split targets, duplicates, and leakage checks are recorded. The ConspiracyBench fixture was not found in this checkout; verify its file and hash before import.
- **Before Phase 5 reads:** freeze the keyword list and rule, character n-gram and classifier settings, software versions, and matched inference precision/kernel/device configuration for parent and candidate. Validate the row-level group-bootstrap intervals and threshold-search coverage. Resample `group_id` values jointly without outcome stratification, carry every row in each selected group, use 10,000 replicates, and seed 0. Report exact one-sided 95% Clopper-Pearson bounds for negative-group any-false-positive and attack-group complete-hit rates as diagnostics only. The interval procedure must handle zero-event bootstrap degeneracy conservatively; an unsupported or degenerate interval cannot grant threshold eligibility or a confirmed pass.
- **Locked test:** open it only after the candidate, temperature, threshold, and interval method are frozen. A confirmed pass requires row-level recall lower confidence bound at least 90% and row-level false-positive upper confidence bound at most 1%. A point-estimate-only pass is provisional. If the test informs a change, retain it as regression and create a fresh locked test for the next version.

The 2,048-token gate includes `<state>`. Preserve raw text in the dataset, while relying on Kev's encoder safeguard for fake special-token spellings. Phase 0 did not download or label source data, build suites, run model reads, train, call paid services, or open locked tests.

## Validation and next step

`registration.json` parsed with PowerShell `ConvertFrom-Json`. The static checks confirmed its schema version, phase status, four-trial grid, Kev-4B scope, pending model lock, and fixed bootstrap convention. The SHA-256 for the user-edited `MASTER-PLAN.md` still matches the pre-existing audit hash. Root's Kev knowledge lookup returned GitHub 404 for the expected `data-and-synthetic.md`; this handoff does not infer prior-session history. No tests were added or run.

The next native-Windows command to reproduce the registration JSON parse is:

```powershell
python -m json.tool experiments/judge-manipulation/registration.json
```

Phase 1 itself has no repository CLI; it begins with the source audit documented in [README.md](README.md). The trainer's `--help` command is not the next step because Kev's suite module imports `fcntl` on import, and the WSL environment is currently stopped.

## Artifact hashes

SHA-256 values below cover the final contents of the listed files. The digest for this handoff is reported with the phase-0 status message because a file cannot include its own final digest.

| File | SHA-256 |
|---|---|
| `MASTER-PLAN.md` | `812f13d43f41e05597b09fd783a7486aa7999f80fd995db94815b826c17892ba` |
| `experiments/judge-manipulation/registration.json` | `e9cdf442f451ac5344aa7d188db285ccb241b4e7ffb0be032efd3041dbe28c76` |
| `experiments/judge-manipulation/README.md` | `96000fac817769ce3fc45bf5ba0430d828b0bca423efbd44d2b846110f3c1c65` |
| `experiments/judge-manipulation/git-state.json` | `3e5dc239275fcc4abc58f1b4837da6152cf57929cce1cbd16e5b4dbbbe9a1554` |
| `experiments/judge-manipulation/git-workflow.md` | `d46dfc3dc1bcb88f2f8a5646f1d06419678eb5ea3553078be4e8d8dd9f81d494` |

The registration directory is `C:/Users/Tomer/Documents/GitHub/kev/experiments/judge-manipulation/`. The dataset destination remains `C:/Users/Tomer/Documents/GitHub/kev/evals/judge-manipulation-v1/` and has not been created.
