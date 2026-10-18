# Phase 5 readiness and launch protocol

This Phase 5 preregistration draft fixes the Kev-4B recipe, the four seed-0 trials, two confirmation seeds, baseline rules, and uncertainty procedure. Root must review and lock it before any baseline read or training. It does not supply corpus rows or establish hardware feasibility. The original resident-VRAM failure remains historical evidence. The amended shared-memory probe must pass its own registered limits before training.

The live gates are in [phase5-readiness-gates.json](phase5-readiness-gates.json). They start pending. After root approves the preregistration method, change its status to `preregistered_blocked` and bind its final SHA-256 in the gate file. The CPU-only checker validates the current draft schema and explicit evidence/review attestations, then exits with code 2 while any approval remains pending:

```powershell
python experiments/judge-manipulation/phase5-readiness-preflight.py
```

The checker audits document structure and hashes; it does not verify model identity, corpus contents, hardware execution, runtime parity, or statistical validity. A root reviewer must inspect the concrete gate-specific evidence and record approval bound to its hashes before marking a gate passed. The corpus gate requires its suite manifest and every partition; the hardware gate requires a concrete Phase 4 hardware run report. These review records are attestations, not automated verification. The Phase 4 shared-memory plan currently says `registered_pending_root_review_and_exclusive_gpu_lease`; it is not a pass. Keep `.agent_harnesses.json`, `.venv`, and the historical Phase 4 records untouched.

## Launch order after all gates pass

Run only from the WSL environment whose dependencies, model lock, data files, and runtime hashes appear in the evidence. First make the three baseline reads on the same frozen train, calibration, and development partitions. Train the character n-gram model on train only. Fit any Kev temperature on calibration only. Select thresholds from development only using the preregistered interval code. Never use locked-test rows for temperature fitting, threshold search, or model choice.

Then run the four seed-0 trials sequentially from the same locked Kev-4B parent and frozen suite. Use separate output directories. Interrupted LoRA runs restart from that parent in a fresh directory; do not resume partial optimizer state. Score each seed-0 model using the same baseline procedure. Select one qualifying trial by the frozen ranking, then record its checkpoint and selection report before running exactly two confirmations with seeds 1 and 2. Confirmations measure stability only.

After the gates pass, run the commands below in the pinned WSL environment. Set `RUN_ID` to a new value and confirm the output directory does not exist. The grid runs sequentially, from the same pinned parent and suite:

```bash
set -euo pipefail
RUN_ID="phase5-$(date -u +%Y%m%dT%H%M%SZ)"
cd /mnt/c/Users/Tomer/Documents/GitHub/kev
PYTHON="$PWD/.venv/bin/python"
KERNEL_OVERLAY="<set to the exact runtime-lock path, or empty if none is registered>"
COMMON=(--suite evals/judge-manipulation-v1 \
  --base Qwen/Qwen3.5-4B-Base \
  --base_revision 1001bb4d826a52d1f399e183466143f4da7b741b \
  --init_from jaredpalmer/kev-4b@139fdd94f1b6a6ad80cc15e08fcb99cac885a101 \
  --weights_dtype bf16 --dtype bf16 --lora 16 --lora_targets all \
  --batch 1 --accum 16 --checkpointing 1 --max_state 2048 --device cuda)
OUT="runs/judge-manipulation-phase5/$RUN_ID"
test -x "$PYTHON"
test -f evals/judge-manipulation-v1/manifest.json
if [ -n "$KERNEL_OVERLAY" ]; then test -d "$KERNEL_OVERLAY"; fi
test ! -e "$OUT"
PYTHONPATH="${KERNEL_OVERLAY:+$KERNEL_OVERLAY:}$PWD${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m kev.train "${COMMON[@]}" --epochs 1 --lr 2e-5 --seed 0 --out "$OUT/lr-2e-5-e1"
PYTHONPATH="${KERNEL_OVERLAY:+$KERNEL_OVERLAY:}$PWD${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m kev.train "${COMMON[@]}" --epochs 2 --lr 2e-5 --seed 0 --out "$OUT/lr-2e-5-e2"
PYTHONPATH="${KERNEL_OVERLAY:+$KERNEL_OVERLAY:}$PWD${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m kev.train "${COMMON[@]}" --epochs 1 --lr 5e-5 --seed 0 --out "$OUT/lr-5e-5-e1"
PYTHONPATH="${KERNEL_OVERLAY:+$KERNEL_OVERLAY:}$PWD${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m kev.train "${COMMON[@]}" --epochs 2 --lr 5e-5 --seed 0 --out "$OUT/lr-5e-5-e2"
```

Use the same `COMMON` array for the two confirmation runs, with the selected learning rate and epoch count, seeds `1` and `2`, and fresh output paths. Before the commands run, verify that `--init_from` resolves the exact locked adapter/head and that the resulting trainable tensors match the registered FP32 recipe. The checker intentionally does not start training.

After development selection, freeze the winning checkpoint hash, recipe, calibration temperature, persisted float threshold, exact statistics implementation hash, and row/group index hashes. Apply the canonical comparator to the preserved probability scalar and persisted threshold. Only then open the locked test, once. Keep its threshold fixed. A point-estimate pass is provisional. If an interval is unsupported or the registered bound does not qualify, the result cannot be called a confirmed pass. If the test informs a change, retain it as regression and make a fresh test set for that version.

## Statistical feasibility and limits

The registered locked test target has 1,000 attacks and 2,000 negatives, but row totals do not establish interval precision when lineage groups are correlated. Phase 3 must report distinct group counts by partition and label, group-size distributions, mixed-label group counts, and any groups lost to strict admission or deduplication. No shortage may be filled by duplicating rows. If independence, the hard group-size cap, or the reviewed source-mixture assumptions are not established, the draft exact bound is unsupported and the associated gate fails closed.

**The interval rule is not approved or launch-ready.** A seeded CPU stress check under several correlated-group and unequal-size synthetic settings found the proposed max-deviation group-bootstrap band covered the full reduced threshold grid in 96%, 93%, 88%, 89%, and 81% of 150 simulated datasets. The pointwise-band-plus-adaptive-selection procedure also under-covered in some settings (89% and 88%); one setting falsely selected a threshold in 9/88 selected datasets. These Monte Carlo checks are deliberately small, use only 20 quantile thresholds rather than every distinct score, and depend on the synthetic random-intercept/logistic model. They reject any claim that the draft bootstrap rule has established 95% coverage; they do not estimate real-corpus coverage. Zero-event boundaries remain unsupported.

An exact finite-sample alternative exists only under a narrower design. The [Neyman–Pearson umbrella rule](https://pmc.ncbi.nlm.nih.gov/articles/PMC5804623/) chooses the cutoff as an order statistic from independent negative calibration units and predicts positive only when `score > cutoff`. Its violation bound is a binomial upper tail; with target FPR .01, the most conservative rank (`k=n`) needs at least 299 independent negatives for one-model 95% confidence. To protect a four-model search by Bonferroni at 95% family-wise confidence, it needs at least 437 independent negatives per candidate threshold frame (`delta=.0125` each). At 437, the valid rank is the maximum score. Ties are safe only with strict `>`; `>=` would include cutoff ties and is outside the guarantee. The score model must be fixed without using those threshold units.

A conservative way to adapt this to clustered rows is possible if Phase 3 enforces and verifies a hard maximum of `M` negative rows per independent group. Calibrate on each group's maximum negative score. For a fixed threshold, `row-FPR = E[false-positive rows] / E[negative rows] <= M * P(any false positive in a group)`. The same order-statistic guarantee on group maxima then gives a conservative row-level upper bound without removing rows. For recall, use each group's minimum attack score; `row-FNR <= M * P(any miss in a group)`. Under IID group sampling from the target mixture, zero any-FP groups require 1,497 negative groups for a 95% row-FPR upper bound <=1% when M=5; the four-trial development search needs 2,189 negative groups by Bonferroni. Zero any-miss groups require 149 attack groups for a 95% row-recall lower bound >=90% when M=5. These are sufficient worst-case collection counts, not acceptance guarantees for a stratified or non-IID source mixture. A fixed source-stratified sample needs valid stratum-specific bounds and prespecified row weights. The helper in [phase5-readiness-statistics.py](phase5-readiness-statistics.py) implements this candidate formula and fails closed until a group design is explicitly approved. Its edge tests pass, but the method remains unapproved and does not validate a corpus or score runtime.

The canonical comparator is `Python float(probability) >= persisted float threshold`. It preserves the probability scalar with `Tensor.item()` and compares it with the numeric threshold loaded from JSON, without casting back to the tensor dtype. CPU checks using real torch float32 and float64 tensors, ties, Python `nextafter`, and JSON round-trips pass. They validate this scalar contract only; Kev's actual score path and saved probabilities still require a root-reviewed runtime parity report.

This does not presently resolve the frozen gate. The MASTER target is row-weighted: choose a development threshold whose row-level FPR upper bound is at most 1%, then require the locked-test row-level FPR upper bound to be at most 1%. Its intervals use attack and answer groups as sampling units, and it requires every eligible School of Reward Hacks and NotInject row in the locked test. Phase 2's source audit reports 105 School records, each with unique `user` value and both candidate `school_of_reward_hacks` and `control` fields; one pair per source record would give at most 105 groups if both fields are established as in-scope detector labels, but provenance alone does not establish those labels. The records also fall within only three task/metric slices. NotInject contains 339 prompts, not submitted responses; there is no artifact establishing eligible negative response rows or unique lineage groups. Preserve those candidates and resolve their eligibility rather than pruning them. A one-row-per-group design across all mandatory external rows is therefore not established, and equal-weight sampling of one row per group would target a group-average rather than the frozen row-average when sizes differ ([estimand discussion](https://pmc.ncbi.nlm.nih.gov/articles/PMC9908044/)). The max-score group bound above may retain the row estimand, but needs a hard group-size bound, enough independent groups, and an IID/stratified inference argument. Until those inputs and the inference rule are reviewed, neither the NP alternative nor the bootstrap draft is approved for threshold selection or acceptance.

The nonparametric bootstrap cannot learn uncertainty for an event it never observes when every resample remains at zero false positives or zero misses. Any later approved rule must mark such a boundary-collapsed interval unsupported. It must not report a zero-width interval as certainty, substitute an independent-row interval, or use the diagnostic group Clopper-Pearson result as the row-level gate. A candidate with no supported qualifying threshold is not selected. A locked-test boundary-degenerate interval makes the outcome provisional.

As a separate group diagnostic, if zero groups contain a false positive, the exact one-sided 95% Clopper-Pearson upper bound is at most 1% only with at least 299 negative groups. If every attack group is fully detected, the corresponding exact lower bound reaches 90% at 29 attack groups. These checks describe the diagnostic group estimands only. They do not prove the row-level gates. The effective group counts are not known yet, so confidence-interval feasibility remains unresolved until the corpus is frozen.

The 1% row-level FPR target is more demanding than an ordinary accuracy check. Until the CI method and row/group design are approved, the development outcome cannot be judged by the draft interval implementation. Do not relax the confidence level, threshold rule, or event handling after seeing scores. Report recall, FPR, precision, macro F1, PR-AUC, confusion matrix, Brier score, NLL, calibration error, source/family/length/placement slices, published/adapted/synthetic slices, latency, throughput, VRAM, host RAM, unsupported inputs, and failures. Mixture precision is not deployment precision.

## Current blockers

- The 200-row Phase 2 disposition has nine accepted rows, and the full pilot gate is not passed.
- No frozen Phase 3 training, calibration, development, or locked-test suite exists yet.
- The original 4B run exceeded the resident 8 GiB limit. The user-approved shared-memory amendment is still awaiting its registered benchmark result.
- Baseline software/kernel locks and the exact statistics implementation hash must be recorded before the first baseline read.
- A validated inference/threshold rule must reconcile the independent-sample exact NP assumptions (including comparator ties and four-trial multiplicity) with the MASTER row-level estimand and mandatory external-row inclusion; this is unresolved.
- The ConspiracyBench fixture remains a compatibility-only input whose current file and hash still need verification before import.

No baseline, full-grid training, confirmation trial, or locked-test read is authorized by this readiness artifact. Root must review the concrete gate reports before launching any real trial.
