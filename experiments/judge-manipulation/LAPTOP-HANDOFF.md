# Laptop handoff

## Latest status (2026-10-18, after the Phase 2 pilot)

The single registered 200-row Phase 2 pilot completed sequentially on the local RTX 4060 Ti. Qwen3-4B Q4_K_M offloaded 37/37 layers; Phi-4-mini Q4_K_M offloaded 33/33 layers. Both produced 200 bound ledger rows with zero transport/inference exceptions; 7 Qwen and 2 Phi responses ended at the pinned length limit and were not retried. The servers were stopped after their passes and the GPU lease is free.

The registered **structural nonvacuity gate passed**, with 5 accepted known-construction rows (3 positive, 2 negative) and 4 accepted OASST agreements. Overall, only 9/200 rows were accepted; 162 were malformed/unsupported, 22 had checker disagreement, and 7 were quarantined for inference truncation. The report found no obvious parser defect. Corpus-scale checker usability, synthetic group-level evidence, and source generalization remain unresolved. This is not a Phase 2 quality pass and does not authorize Phase 3 freeze or training.

The aggregate-only retained report is [here](pilot-reports/2026-10-18/phase2-pilot-report.md), with its machine-readable bindings and hashes [here](pilot-reports/2026-10-18/phase2-pilot-report.json). The accepted semantic rows passed the real Kev loader and encoder. Raw candidate/output ledgers and live server logs remain ignored and outside Git. Continue with the root-coordinated Phase 4 shared-memory source checkpoint and preflight; any GPU benchmark must wait for that coordination. Review the usability and group-coverage findings before proposing further Phase 2 work. Do not retry truncated rows or change the frozen prompt, labels, runtime, or protocol.

Resumed state (2026-10-18): The user has resumed work after the dated credit-stop note in [RESUME-2026-10-18.md](RESUME-2026-10-18.md); that note remains a historical record. New supervisors use GPT-6-Luna Extra High. Prefer one active implementation worker and reuse the existing caches and proofs. `/root/phase_4_luna_director` is finishing the existing shared-memory WIP in the isolated `codex/judge-manipulation-phase-4-shared-memory` worktree at `d716d0f`; its benchmark has not run. The Phase 2 pilot is complete as summarized below. The earlier source checkpoint, focused test evidence, and recovered-GPU evidence remain valid; no new training or quality pass is claimed.

The current work is Phase 2 on `codex/judge-manipulation-phase-2-pilot`, continued on the Windows desktop. The approved [pilot plan](pilot-plan-phase-2.json) specifies 160 programmatic construction rows and 40 pinned OASST English assistant candidates reserved for calibration. The reviewed source/runtime checkpoint is recorded on this branch and the shared progress branch; `SOURCE-CHECKPOINT-phase-2.json` has SHA-256 `a4b2c38e2d248a65d3cc78a587d54a8a05e6d835d55aa6869e951dbfb4fcf34e` and binds 13 artifact hashes. The focused tests reported 24 passes, followed by one separate OASST loader regression pass. The final portable `prepare --parent-tokenizer` command has not been rerun, and no combined 25-test or full fast-suite pass is claimed. The two-checker pilot was still pending at this historical source-checkpoint stage; see the latest status above. No final corpus freeze, candidate training, or locked-test result is claimed.

The user approved keeping BF16 and allowing Windows shared system RAM for a separately registered speed and reliability benchmark. [AMENDMENT-shared-memory-phase-4.json](AMENDMENT-shared-memory-phase-4.json) records that decision. It does not change the old strict-memory failure, the model lock, precision, or 2,048-state-token limit. The amended benchmark requires its own implementation and evidence. The historical strict runner below is not the upcoming amended benchmark command.

The laptop is a MacBook Air M5 with 24 GB memory. Current compute remains on the Windows desktop; the laptop can receive pushed source and documentation. GPU work is coordinated so checker inference and any later training probe run sequentially.

The shared continuation branch is `codex/judge-manipulation-progress` in [tomerzipori/kev](https://github.com/tomerzipori/kev/tree/codex/judge-manipulation-progress). Phase 0 registration is complete at `bb08c11`. Phase 1 source research is complete at `c05fcc3`. Compatibility commit `a2e15a1` preserves the exact `MASTER-PLAN.md` bytes and frozen artifact hashes across Windows checkouts. The repository's `main` branch remains at `925e9e9` and is unchanged.

Phase 4 Windows/WSL feasibility is recorded on the [Phase 4 branch](https://github.com/tomerzipori/kev/tree/codex/judge-manipulation-phase-4-windows-feasibility) and the shared [progress branch](https://github.com/tomerzipori/kev/tree/codex/judge-manipulation-progress). The [Phase 4 handoff](HANDOFF-phase-4.md) and [feasibility gate](feasibility-gate-phase-4.md) contain the full evidence. The registered resident-memory gate failed: peak CUDA allocation reached 10,424,460,288 bytes against a limit of 8,585,216,000 bytes, exceeding it by 1,839,244,288 bytes. The root-directed SIGINT stopped the trainer with `KeyboardInterrupt`; no CUDA OOM occurred. The run produced no evidence of an optimizer update or a completed checkpoint. At that historical handoff point, Phase 2's 200-row pilot and Phase 3's frozen suite build were still pending; the pilot is now recorded above, while Phase 3 remains blocked pending review of its limitations. A separate amended shared-memory probe is registered on the [Phase 4 shared-memory branch](https://github.com/tomerzipori/kev/tree/codex/judge-manipulation-phase-4-shared-memory) at commit `d716d0f41807c501b5987d99cc6c6f361f6ea1fd`. Its CPU overlay is prepared; GPU ABI/kernel correctness and training have not run. The probe waits for an exclusive GPU lease after Phase 2 releases the model and does not alter the historical resident-VRAM failure.

The retained [raw report JSON](report-phase-4.json) and [raw report Markdown](report-phase-4.md) record the attempt. The first full fast-suite run reported 475 passed, 26 skipped, 1 failed, and 2 errors because three tests could not write to the root-owned default Hugging Face cache. The three affected tests passed when rerun with the configured cache. The full suite was not rerun. Python compilation passed for the five Phase 4 scripts, and `bash -n` passed for the WSL setup script. These checks do not establish a full-suite pass.

The following command reproduces the historical strict-memory probe with the locked environment and fresh output and report directories. It does not implement the shared-RAM amendment:

```sh
bash experiments/judge-manipulation/setup_phase4_wsl.sh
RUN_ID="phase4-$(date -u +%Y%m%dT%H%M%SZ)"
.venv/bin/python experiments/judge-manipulation/run_phase4_smoke.py \
  --out "runs/judge-manipulation-phase4/$RUN_ID" \
  --report-dir "experiments/judge-manipulation/reports/$RUN_ID"
```

Use a new `RUN_ID` for each attempt. The script refuses to overwrite an existing output directory or report files. Review generated reports and logs before committing them.

To replay only the recorded memory assessment from the raw report, run this from the repository root. It loads no model and starts no GPU process:

```sh
.venv/bin/python - <<'PY'
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory

script = Path("experiments/judge-manipulation/run_phase4_smoke.py")
spec = importlib.util.spec_from_file_location("phase4_smoke", script)
phase4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase4)
report = json.loads(Path("experiments/judge-manipulation/report-phase-4.json").read_text(encoding="utf-8"))
with TemporaryDirectory(prefix="phase4-report-replay-") as runtime:
    result = phase4.memory_budget_assessment(report, Path(runtime))
print(json.dumps(result, indent=2, sort_keys=True))
PY
```

To start from a fresh laptop checkout, run:

```sh
git clone --branch codex/judge-manipulation-progress git@github.com:tomerzipori/kev.git
cd kev
```

In an existing clone, fetch and create a local branch that tracks the shared branch:

```sh
git fetch origin
git switch --track origin/codex/judge-manipulation-progress
```

To update an existing local progress branch, run:

```sh
git fetch origin
git switch codex/judge-manipulation-progress
git pull --ff-only origin codex/judge-manipulation-progress
```

Before each later push, fetch and check whether the shared branch moved:

```sh
git fetch origin
git status --short --branch
git log --left-right --oneline HEAD...origin/codex/judge-manipulation-progress
```

If the remote has commits that are missing locally, integrate them before pushing. Use a normal fast-forward pull when possible, or resolve and rebase local work as needed. Never force-push over laptop work.

Model caches, `.venv`, raw inputs, weights, and other large local files are excluded from Git. They will not appear on the laptop. Follow the setup instructions and source manifests to recreate dependencies and obtain permitted inputs.
