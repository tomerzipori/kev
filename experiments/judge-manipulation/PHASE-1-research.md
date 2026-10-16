# Phase 1 source research

The source-audit gate passes with exclusions. The registry covers the six registered sources and three additions, with immutable repository revisions, inspected schemas and count units, artifact hashes, label meanings, reuse evidence, lineage advice, and partition restrictions. A source that lacks access, rights, or compatible labels remains excluded. This result does not establish corpus sufficiency, final labels, or model feasibility.

[source-registry.json](source-registry.json) is the combined record. [audit-shortlist.json](audit-shortlist.json) and [audit-attacks.json](audit-attacks.json) retain the detailed evidence from the two research workers and director verification. No frozen Phase 0 document changed.

## Source decisions

| Source | Inspected evidence and decision |
|---|---|
| School of Reward Hacks | The pinned CSV has 1,073 source records. The three judge-oriented task groups contain 44, 44, and 17 records, totaling 105 candidate pairs. Each pair has a hack response and control, but those roles are not detector labels. Reserve the whole source from training, calibration, and development. Only task-compatible candidates may later enter the locked test. |
| NotInject | There are 339 benign source prompts, with 113 in each trigger-word subset. These are user prompts rather than assistant responses. The Multilingual category has 84 rows, which is not an exact non-English count. Reserve all 339 from training, calibration, and development; decide English response-style versus transfer/report-only eligibility before freezing the test. |
| JudgeDeceiver | The two inspected basic suffix files each have 10 benchmark question records with model-specific suffix outputs. No reuse license was found. Exclude. Suffix variants cannot be counted as independent attacks. |
| HackAPrompt | The card declares MIT, but the pinned Parquet is gated. An existing local credential could not access its metadata, and no contact-sharing agreement was accepted. The public tree masks its LFS digest. File count and contents remain unverified. Exclude pending access, operative terms, content hash, and schema/count inspection. |
| OASST1 | Both pinned Parquets matched their LFS SHA-256 values and parsed successfully. They contain 88,838 messages, including 25,326 English assistant messages. These counts precede deletion/review checks, target labels, and token admission. Apache-2.0 is documented in the card and license. Root approved reserving the whole source for calibration candidates. |
| deepset prompt-injections | The 662 rows have generic injection labels. The pinned card says CC-BY-4.0 inside dataset_info and Apache-2.0 at the top level. The pinned main LICENSE URL returns 404. A separate child commit adds an Apache license but does not resolve the conflicting card. Exclude. |
| EvalHack | The paper describes 1,000 base answers with clean and attacked variants. The committed JSON actually contains 220 base-answer objects, plus transformation code. No data reuse license was found. Exclude; do not treat the paper protocol as a released labeled attack corpus. |
| Peer-review indirect injection | The repo has 100 selected-paper rows and five attack-family prompt definitions. These are documents/templates, not labeled submitted-response attacks. Reuse permission was not established. Exclude for rights and scope. |
| TensorTrust | The inspected 230-row extraction-detection file contains model outputs and extraction labels, not attacker inputs. Its large raw dump was not parsed. Site consent describes public research release, but dataset adaptation/reuse terms remain unclear. The separate game's code license does not resolve data permission. Exclude. |

The registry links each claim to pinned primary artifacts. The main source corrections are visible in the [School card](https://huggingface.co/datasets/longtermrisk/school-of-reward-hacks/blob/d7e04a550119cb5410494cf90e2313284a5f2148/README.md), [NotInject card](https://huggingface.co/datasets/leolee99/NotInject/blob/847ae76cf8fea5ed325429e569ae8cfef022d2e0/README.md), [deepset card](https://huggingface.co/datasets/deepset/prompt-injections/blob/4f61ecb038e9c3fb77e21034b22511b523772cdd/README.md), and [OASST1 license](https://huggingface.co/datasets/OpenAssistant/oasst1/blob/fdf72ae0827c1cda404aff25b6603abec9e3399b/LICENSE).

## Recommended path to the pilot

Root approved OASST1 as a wholly reserved calibration body source in place of the excluded deepset route. Calibration attacks would require separately frozen, known judge-focused constructions and calibration-only seed/template families. Keep every body, conversation, attack, control, and derived variant together. Existing OASST review/rank/toxicity fields cannot supply the target label.

Training can use new source-free constructions under the registered rule. HackAPrompt-derived material remains unavailable until its separate access and verification review passes. No newly audited independent answer-side attack corpus was admitted. The registered collection targets remain goals; these source counts do not show that those goals can be met. Do not repeat rows or expand suffixes to claim a larger independent sample.

The 200-row pilot must prove label mapping, known-construction evidence, checker quarantine, and lineage rules before full processing. It must resolve School target-label disposition and NotInject scope without tuning against final test performance. Exact/near duplicates and cross-source lineage have not yet been checked. The model lock, Windows smoke, statistical implementation, and missing ConspiracyBench fixture remain separate gates.

## Validation boundary

The director independently reproduced School's CSV count and judge-group counts, inspected the conflicting deepset card, parsed both pinned OASST Parquets, and matched their downloaded bytes to the recorded LFS hashes. The source reports record other artifact inspections and limitations. Raw verification files are in system temporary storage, outside the repository.

Public source text, schema, and counts were inspected. No claim is made that all future test text was unseen. No model scoring, threshold tuning, final locked-test evaluation, pilot, corpus, target labels, paid API, GPU job, Git mutation, commit, publication, or deployment occurred in this phase. No tests or production code were added.

The next runnable validation command is:

```powershell
python -m json.tool experiments/judge-manipulation/source-registry.json
```

Root approved this source-audit gate and the OASST calibration reservation on 2026-10-15. The pilot and model gates remain pending. This handoff does not start Phase 2.
