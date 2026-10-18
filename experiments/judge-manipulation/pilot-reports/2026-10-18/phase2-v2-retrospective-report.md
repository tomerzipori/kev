# Phase 2 v2 retrospective validation

This is an aggregate-only policy comparison over the exact frozen v1 candidates and bound outputs. It is not a fresh pilot, does not pass the v2 pilot gate, and does not authorize bulk generation, training, or Phase 3 freeze. The v1 disposition ledger and all model outputs remain unchanged.

The new v2 rule admits known constructions only after exact deterministic reconstruction from the frozen lock, exact candidate/label/lineage/request matching, and recomputed evidence offsets and hashes. The two model outputs remain QC and cannot overwrite or veto those provenance-backed labels. OASST remains unlabeled semantic data and still requires two complete supported checks with exact agreement and evidence.

The retrospective reconstruction verified 160 of 160 known rows: 80 positive and 80 negative. It accepts four OASST agreements, quarantines 29 malformed or unsupported OASST outputs, and quarantines seven inference failures. Total retrospective acceptance is 164/200. Construction QC still records 22 disagreements; this does not alter the 160 verified labels.

| Checker | Valid supported | Malformed | Valid unsupported | Incomplete |
| --- | ---: | ---: | ---: | ---: |
| Qwen 3.5 4B Q4_K_M | 146 | 13 | 1 | 0 |
| Phi-4 Mini Q4_K_M | 29 | 73 | 57 | 1 |

The construction lock covers 80 matched pairs across eight attack classes (10 each), five placements (16 each), six negative controls (13–14 each), 24 template families, 80 body keys, and 80 attack-seed keys. The 80 pairs reuse only three body recipes and three filler recipes (24, 24, and 32 pairs per recipe). They remain three effective synthetic groups: one each for train (64 rows), calibration (48), and development (48). Keep these groups intact; this design is not adequate evidence of broad independent coverage.

The pinned School of Reward Hacks audit has a 105-pair-only upper bound. Joining pairs that share an exact repeated 64-character suffix hash in either response, followed by transitive closure, yields 98 components: 93 singletons, three pairs, and two triples. This is a conservative grouping cap, not proof of independence or label eligibility. NotInject contributes 339 unique prompt groups, but no submitted assistant responses eligible for negative labels. Additional independent in-scope response roots are required.

Both checkers have smoke-only CUDA offload evidence: Qwen 37/37 layers, Phi 33/33. Qwen's smoke reached the pinned-model and JSON assertions but its final report has a mounted-log I/O gap, and no raw completion body was persisted. Phi's generic smoke passed. These one-request smokes did not call the pilot checker prompts and establish no full pilot readiness. The GPU lease is released.

Frozen input hashes:

| Input | SHA-256 |
| --- | --- |
| MASTER-PLAN.md | `812f13d43f41e05597b09fd783a7486aa7999f80fd995db94815b826c17892ba` |
| source registry | `2818e4f68a8adc352d71112b195eb5be7b389b47796ce3c6e63f80d0efb19ec7` |
| pilot plan | `8f870598a6463868ddfd7254b0682027e1d06d02845c6e73c93dd66809316fb0` |
| construction lock | `e247737c09c304495cde56194a5f82fc7ebbf51e9610652197903dbbded8cf33` |
| prompt | `ec0c7c03afb8938f4466457914df45efa738affb87a52887517ea325ec95b505` |
| runtime lock | `1a767ebd93ce48ab0dd4e1e0490a01bf76d84f6899e64327cb4460f02f6fe720` |
| candidates | `bcb35908610e50c3e2cc1c787d1af1118735bcd67b9373af17ed595c1252939f` |
| Qwen ledger | `d7dd33ba85723d2f49af7a0760802cc2c921273d0c8b34e06ec6cfa220207a25` |
| Phi ledger | `79e1c4a8836050f9a404d8b408d5aa13d2d2ae698f7a4c82e68dc14a77c7456e` |
| v1 dispositions | `c3b924e4e7d6e9b8da54af40f6ba8817776e7d28524d4895a0ad98fc479c9f0d` |
| v1 aggregate report | `8549d568af49df9883e962fdca4dd7c27eead20314bdfe50f9ba8d83eaf80340` |

The JSON report contains the runtime proof hashes, source-audit hash, accepted-OASST loader hash, checker QC totals, and full coverage counts. The focused regression test passes on CPU: `2 passed`.
