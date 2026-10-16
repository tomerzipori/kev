# Phase 1 handoff

The Phase 1 source-audit gate passes with exclusions. Root approved the evidence and OASST calibration reservation on 2026-10-15. No Phase 2 work started.

The registry audits all six registered sources and three additions. It records immutable revisions, schemas, count units, label meanings, provenance, artifact hashes or explicit unavailable-digest limits, reuse evidence, partition restrictions, and lineage recommendations. Unknown or incompatible sources remain excluded, so the audit does not grant permission to ingest their text.

## Completed work

- Created [source-registry.json](source-registry.json) as the combined pinned record and [PHASE-1-research.md](PHASE-1-research.md) as its research summary. Detailed evidence remains in [audit-shortlist.json](audit-shortlist.json) and [audit-attacks.json](audit-attacks.json).
- Corrected the source-count interpretation without rewriting the frozen plan. School has 1,073 source records, of which 105 belong to the three judge-oriented task groups. Those 105 are paired response candidates, not 105 target labels or 210 admitted examples.
- Reserved all School and NotInject material from training, calibration, and development. NotInject has 339 benign source prompts. Its 84 Multilingual-category rows do not determine the exact non-English count, and submitted-response eligibility remains open.
- Excluded JudgeDeceiver, deepset, EvalHack, peer-review indirect injection, and TensorTrust for the documented rights, scope, provenance, or framework/data limits. No added independent answer-side attack corpus was admitted.
- Excluded HackAPrompt pending its gated-file access, terms, content hash, schema, and count verification. An existing local Hugging Face credential returned GatedRepoError for the pinned file's metadata. Connected HF metadata also reports gated. No contact-sharing agreement was accepted. Public metadata masks its LFS/Xet digest, and the displayed Git blob SHA-1 is not a data SHA-256.
- Verified both OASST1 pinned Parquets against their LFS SHA-256 values and parsed their schemas/counts. They contain 88,838 messages, including 25,326 English assistant messages before review/deletion, target-label, and token-admission checks. Root approved reserving OASST1 wholly for calibration candidates under its documented Apache-2.0 terms. This replaces the excluded deepset route.

The approved source-level calibration policy requires calibration-only construction seed/template families and source conversations held from training. It does not assign negatives to ordinary OASST messages. All derived bodies, attacks, controls, and variants must stay with their source group. Until HackAPrompt is separately cleared, the training path consists of new source-free constructions under the registered rule. Target counts and coverage are unproven.

## Validation

All three JSON artifacts parse. Static checks verified nine unique source IDs, immutable 40-character revision digests, primary URLs and artifact entries for every source, and valid format for all recorded SHA-256 values. Audit-file digests match the combined registry. Missing gated/raw-dump content hashes are explicit limitations attached to excluded sources.

The director independently reproduced School's CSV total and exact 44/44/17 judge-group counts, inspected the conflicting deepset card bytes, and downloaded, hash-checked, and parsed the two pinned OASST Parquets with PyArrow. The shortlist worker also verified all five NotInject/deepset Parquet downloads against their LFS digests. Raw downloads remain in system temporary storage outside the repository.

The frozen MASTER-PLAN, registration, README, Phase 0 handoff, git-state, and git-workflow hashes all match the phase boundary. No production code or tests changed, and no model, baseline, paid API, GPU job, target labeling, pilot, corpus construction, threshold tuning, or final locked-test evaluation occurred. Public source text inspection did occur; this handoff does not claim all future test text was unseen. The source researchers performed no Git mutations, commits, publication, or deployment.

## Unresolved work and next command

No unresolved item permits silent source reuse. deepset's current card conflict remains unresolved and the source stays excluded. HackAPrompt remains inaccessible through the checked existing credential. The added sources do not supply a licensed independent attack population. All corpus targets remain collection goals.

Before full processing, the required 200-row pilot must validate task-label mappings, construction evidence, checker quarantine, and connected lineage groups. School target-label disposition and NotInject language/response-role disposition remain pending. Implement the approved calibration source reservation and freeze separate construction families before making pilot rows. Exact/near duplicate checks, strict token admission, final eligible counts, and leakage checks belong to later gates.

The Kev-4B parent/model lock, actual WSL2 training feasibility, statistical implementation, baseline freeze, and ConspiracyBench fixture verification also remain pending. Phase 1 establishes none of these.

The next native Windows command validates the combined registry:

```powershell
python -m json.tool experiments/judge-manipulation/source-registry.json
```

## Artifact hashes

SHA-256 values cover final file bytes. This handoff's own digest is reported separately to root after it is written.

| File | SHA-256 |
|---|---|
| `source-registry.json` | `2818e4f68a8adc352d71112b195eb5be7b389b47796ce3c6e63f80d0efb19ec7` |
| `audit-shortlist.json` | `be94552acafcebc9b2d3c7c342549619b320b52eb8315aa93fa201bf81b3d4a0` |
| `audit-attacks.json` | `8788e7f94865fab559def10b6ac6ea6de005214a7b2e6d7418193e611058384b` |
| `PHASE-1-research.md` | `49efff3913135c29ecb95acc8114d965a7495da22d3d9905ad0b334bf38a1806` |
| `BRANCH-phase-1.json` | `967bfcb94416cdf4cc13f1bda5a6975177a0c896684a86ac7a530fb44aa1a264` |
| `MASTER-PLAN.md` | `812f13d43f41e05597b09fd783a7486aa7999f80fd995db94815b826c17892ba` |
| `registration.json` | `e9cdf442f451ac5344aa7d188db285ccb241b4e7ffb0be032efd3041dbe28c76` |
| `README.md` | `96000fac817769ce3fc45bf5ba0430d828b0bca423efbd44d2b846110f3c1c65` |
| `HANDOFF-phase-0.md` | `8882d3628db351447a08b3efb3d8b190607668fb1066d935aae3ad8c2cef0a6a` |
| `git-state.json` | `3e5dc239275fcc4abc58f1b4837da6152cf57929cce1cbd16e5b4dbbbe9a1554` |
| `git-workflow.md` | `d46dfc3dc1bcb88f2f8a5646f1d06419678eb5ea3553078be4e8d8dd9f81d494` |
