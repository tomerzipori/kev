# Laptop handoff

The shared continuation branch is `codex/judge-manipulation-progress` in [tomerzipori/kev](https://github.com/tomerzipori/kev/tree/codex/judge-manipulation-progress). Phase 0 registration is complete at `bb08c11`. Phase 1 source research is complete at `c05fcc3`. Compatibility commit `a2e15a1` preserves the exact `MASTER-PLAN.md` bytes and frozen artifact hashes across Windows checkouts. The repository's `main` branch remains at `925e9e9` and is unchanged.

Phase 4 Windows feasibility started early, alongside source research as the plan allows. It is still in progress. Phase 2's 200-row pilot and Phase 3's frozen suite build have not passed. The current remote continuation branch does not yet include the ongoing Phase 4 scripts or a measured result. The root will update this handoff when the Phase 4 result is ready.

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
