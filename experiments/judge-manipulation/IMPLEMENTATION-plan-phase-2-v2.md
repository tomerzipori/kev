# Phase 2 v2 amendment and readiness plan

The v1 pilot admitted 9 of 200 rows because `phase2_checker.py` asked two models to validate labels already established by the deterministic construction. The frozen v1 plan required that quarantine rule, so v1 dispositions stay exactly as recorded. The authorized v2 repair changes label authority for known constructions only; OASST remains semantic and still requires two valid agreeing checks.

`phase2_checker_v2.py` verifies a construction by rebuilding it from the pinned lock, comparing its state, label, lineage, request, and Kev record, and recomputing the exact evidence occurrence and hashes. It then accepts the deterministic label. Checker outcomes remain separate QC. OASST continues through the existing v1 two-checker semantic path. The prompt, runtime, model pins, and v1 outputs are unchanged.

The retrospective application used only the exact v1 candidate file and both bound 200-record ledgers. It verified all 160 construction rows, accepted 80 positives and 80 negatives from their construction labels, and retained 4 OASST semantic agreements. It leaves 29 OASST rows malformed or unsupported and 7 failed. The 164 accepted total is a retrospective policy comparison, not a fresh v2 pilot pass. The v1 disposition ledger, aggregate report, prompt, runtime, and raw model ledgers remain unchanged.

The construction coverage is still too narrow for corpus readiness. The frozen lock has 80 matched pairs across 8 attack classes, 5 placements, and 6 negative-control types; it uses 24 wording families, 80 body keys, and 80 attack seeds. The 80 pairs distribute over only 3 body recipe keys and 3 filler recipe keys, one per reservation. All generated rows in each reservation therefore remain one effective lineage group. Keep those 3 groups. Do not weaken their lineage or create cosmetic IDs.

Before bulk construction, audit whether distinct answer roots and seeds are independent despite shared recipes. If recipe reuse creates a meaningful common lineage, count it as a cluster and report the shortage. Expand independent body and filler recipes across train, calibration, development, and locked test; retain all eight attack classes, five placements, and six controls as design factors. Report group counts by split and label, body/wording/filler family slices, and held-family-out performance. The pinned School audit has a pair-only upper bound of 105, conservatively reduced to 98 transitive components by repeated 64-character response-suffix linkage; the 339 NotInject records are prompt-only. Neither source currently establishes eligible locked-test detector rows or true independence.

The frozen row targets remain 12,000 total and, for locked test, 1,000 attacks plus 2,000 negatives. The earlier M=5 group floors (including 400 negative-bearing and 300 attack-bearing locked-test groups) and zero-event bootstrap diagnostics are historical and rejected; they do not support the frozen row-level 95% FPR or recall targets. Phase 5's exact group-maximum bound and conditional collection counts are a draft only. The Phase 5 readiness registration and [readiness analysis](phase5-readiness.md) show that the prior bootstrap stress procedure undercovers in tested scenarios and leave the exact-bound assumptions unresolved. No group collection counts or inference method are approved here. Resolve actual lineage, mandatory source eligibility, source mixture, sampling assumptions, and hard cap enforcement, then obtain root review before setting counts or using any bound for selection or acceptance.

Phase 1 admitted no compatible, independently reusable attack corpus. OASST's 25,326 English assistant messages are an unfiltered source count, not target labels; the retrospective pilot has only 4 accepted OASST rows, all negative. Keep OASST calibration-only and require two valid agreeing checks for every new semantic label. Do not infer attack labels from OASST metadata. The pinned School of Reward Hacks CSV has 105 pair-only candidate records, but a conservative transitive union by exact repeated 64-character suffix hashes joins them into 98 components (93 singletons, three pairs, two triples; 12 pairs touched). This is a conditional capacity ceiling, not an independence or target-eligibility finding; source IDs and seeds are absent. Its three task/metric families remain coverage slices, not automatic lineage IDs. Pinned NotInject has 339 unique user-prompt groups but no submitted assistant responses eligible as negatives. These sources do not establish the 12,000-row target or balanced, diverse semantic coverage.

No new rows, model calls, full-corpus processing, Phase 3 freeze, or training are part of this change. Root review is required before a fresh v2 pilot or any of those later steps. No GPU lease was used.

The focused regression test is `tests/test_phase2_checker_v2.py`; it runs on CPU without weights, a server, network access, or tokenizer downloads. The main CI workflow previously omitted all tests under `experiments/`, and the broader Phase 2 experiment test file includes pinned-tokenizer checks. CI now adds only the focused, self-contained v2 regression test. Run it locally with:

```bash
.venv/bin/python -m pytest tests/test_phase2_checker_v2.py -q
```

The retrospective audit command is CPU-only and verifies frozen hashes before writing an aggregate-only report:

```bash
.venv/bin/python experiments/judge-manipulation/phase2_checker_v2.py
```

This prepares the protocol and retains the evidence needed for review. It does not establish full-corpus quality, group sufficiency, model performance, or training readiness.
