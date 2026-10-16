# Laptop handoff

The shared continuation branch is `codex/judge-manipulation-progress` in [tomerzipori/kev](https://github.com/tomerzipori/kev/tree/codex/judge-manipulation-progress). Phase 0 registration is complete at `bb08c11`. Phase 1 source research is complete at `c05fcc3`. Compatibility commit `a2e15a1` preserves the exact `MASTER-PLAN.md` bytes and frozen artifact hashes across Windows checkouts. The repository's `main` branch remains at `925e9e9` and is unchanged.

Phase 4 Windows/WSL feasibility is recorded on the [Phase 4 branch](https://github.com/tomerzipori/kev/tree/codex/judge-manipulation-phase-4-windows-feasibility) and the shared [progress branch](https://github.com/tomerzipori/kev/tree/codex/judge-manipulation-progress). The [Phase 4 handoff](HANDOFF-phase-4.md) and [feasibility gate](feasibility-gate-phase-4.md) contain the full evidence. The registered resident-memory gate failed: peak CUDA allocation reached 10,424,460,288 bytes against a limit of 8,585,216,000 bytes, exceeding it by 1,839,244,288 bytes. The root-directed SIGINT stopped the trainer with `KeyboardInterrupt`; no CUDA OOM occurred. The run produced no evidence of an optimizer update or a completed checkpoint. Phase 2's 200-row pilot and Phase 3's frozen suite build remain pending.

The retained [raw report JSON](report-phase-4.json) and [raw report Markdown](report-phase-4.md) record the attempt. The first full fast-suite run reported 475 passed, 26 skipped, 1 failed, and 2 errors because three tests could not write to the root-owned default Hugging Face cache. The three affected tests passed when rerun with the configured cache. The full suite was not rerun. Python compilation passed for the five Phase 4 scripts, and `bash -n` passed for the WSL setup script. These checks do not establish a full-suite pass.

For a later GPU retest after review of the failed gate, use the locked environment and fresh output and report directories:

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
