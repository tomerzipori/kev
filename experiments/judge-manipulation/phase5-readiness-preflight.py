"""Audit Phase 5 registration and root-review evidence attestations.

This checks document shape and file hashes only. It does not verify model
identity, corpus semantics, hardware results, runtime parity, or method validity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRATION = Path("experiments/judge-manipulation/phase5-readiness-registration.json")
DEFAULT_GATES = Path("experiments/judge-manipulation/phase5-readiness-gates.json")
REQUIRED_GATES = (
    "model_lock",
    "preregistered_selection_review",
    "phase2_pilot",
    "phase3_frozen_corpus",
    "amended_phase4_hardware",
    "runtime_and_interval_validation",
)
EXPECTED_GRID = {
    (0.00002, 1, 0),
    (0.00002, 2, 0),
    (0.00005, 1, 0),
    (0.00005, 2, 0),
}
EXPECTED_COMPARATOR = "Python float(probability) >= persisted float threshold"
EXPECTED_METHOD = "group_max_order_statistic_with_exact_binomial_bounds"
GATE_EVIDENCE_KINDS = {
    "model_lock": {"model_lock"},
    "preregistered_selection_review": {"root_review"},
    "phase2_pilot": {"pilot_disposition"},
    "phase3_frozen_corpus": {
        "suite_manifest",
        "train_partition",
        "calibration_partition",
        "development_partition",
        "test_partition",
    },
    "amended_phase4_hardware": {"hardware_run_report"},
    "runtime_and_interval_validation": {
        "runtime_parity_report",
        "interval_validation_report",
    },
}
EXACT_EVIDENCE_PATHS = {
    ("model_lock", "model_lock"): "experiments/judge-manipulation/model-lock-phase-4.json",
    ("preregistered_selection_review", "root_review"): "experiments/judge-manipulation/phase5-readiness-review.json",
    ("phase2_pilot", "pilot_disposition"): "experiments/judge-manipulation/pilot-reports/2026-10-18/phase2-pilot-report.json",
    ("phase3_frozen_corpus", "suite_manifest"): "evals/judge-manipulation-v1/manifest.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_file(relative: str) -> Path:
    candidate = (ROOT / relative).resolve()
    if not candidate.is_relative_to(ROOT):
        raise ValueError(f"path escapes repository: {relative}")
    return candidate


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def validate_registration(registration: dict) -> list[str]:
    failures: list[str] = []
    if registration.get("phase") != 5:
        failures.append("registration phase must be 5")
    if registration.get("status") != "preregistered_blocked":
        failures.append("registration must have a root-reviewed preregistered_blocked status before launch")
    grid = registration.get("trial_grid", [])
    actual = {
        (row.get("learning_rate"), row.get("epochs"), row.get("seed"))
        for row in grid
        if isinstance(row, dict)
    }
    if len(grid) != 4 or actual != EXPECTED_GRID:
        failures.append("seed-0 grid must contain exactly the four registered LR/epoch pairs")
    if registration.get("confirmation_seeds") != [1, 2]:
        failures.append("confirmation seeds must be exactly [1, 2]")
    if registration.get("baselines", {}).get("character_ngram_logistic_regression", {}).get("training_partition") != "training only":
        failures.append("character n-gram model must train on the training partition only")
    if registration.get("probability_and_decision", {}).get("temperature_fit_partition") != "calibration only":
        failures.append("temperature fitting must be restricted to calibration")
    if registration.get("probability_and_decision", {}).get("threshold_partition") != "development only":
        failures.append("threshold selection must be restricted to development")
    stats = registration.get("uncertainty", {})
    if stats.get("status") != "approved":
        failures.append("uncertainty method is draft/unapproved")
    if stats.get("method") != EXPECTED_METHOD:
        failures.append("uncertainty method must be the reviewed group-max exact-bound method")
    if stats.get("comparator") != EXPECTED_COMPARATOR:
        failures.append("probability comparator must preserve the Python float value and compare against the persisted float threshold")
    if not stats.get("assumptions") or not isinstance(stats.get("assumptions"), list):
        failures.append("uncertainty method must state its sampling and group-size assumptions")
    return failures


def _validate_gate_evidence(name: str, entry: dict, failures: list[str]) -> None:
    evidence = entry.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        failures.append(f"gate {name} has no typed concrete evidence")
        return
    kinds: set[str] = set()
    evidence_hashes: dict[str, str] = {}
    for item in evidence:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            failures.append(f"gate {name} contains malformed evidence")
            continue
        path = repo_file(item["path"])
        kind = item.get("kind")
        if not isinstance(kind, str) or kind not in GATE_EVIDENCE_KINDS[name]:
            failures.append(f"gate {name} has unexpected evidence kind: {kind}")
        else:
            kinds.add(kind)
        exact_path = EXACT_EVIDENCE_PATHS.get((name, kind))
        if exact_path is not None and item["path"] != exact_path:
            failures.append(f"gate {name} {kind} evidence must be {exact_path}")
        if name == "phase3_frozen_corpus" and isinstance(kind, str) and kind.endswith("_partition"):
            if not item["path"].startswith("evals/judge-manipulation-v1/"):
                failures.append(f"gate {name} partition evidence must be inside the frozen suite")
        if name == "amended_phase4_hardware" and not (
            item["path"].startswith("experiments/judge-manipulation/")
            and "phase4" in item["path"].lower()
            and "report" in Path(item["path"]).name.lower()
        ):
            failures.append(f"gate {name} requires a concrete Phase 4 hardware run report")
        if not path.is_file():
            failures.append(f"gate {name} evidence is missing: {item['path']}")
        elif item.get("sha256") != sha256(path):
            failures.append(f"gate {name} evidence hash changed: {item['path']}")
        else:
            evidence_hashes[item["path"]] = item["sha256"]
    missing_kinds = GATE_EVIDENCE_KINDS[name] - kinds
    if missing_kinds:
        failures.append(f"gate {name} lacks required evidence kinds: {', '.join(sorted(missing_kinds))}")

    review = entry.get("root_review")
    if not isinstance(review, dict) or review.get("reviewer") != "root" or review.get("decision") != "approved":
        failures.append(f"gate {name} requires an explicit root-reviewed approval record")
        return
    review_path = review.get("path")
    if not isinstance(review_path, str):
        failures.append(f"gate {name} root review record has no path")
        return
    path = repo_file(review_path)
    if not path.is_file() or review.get("sha256") != sha256(path):
        failures.append(f"gate {name} root review record is missing or hash-mismatched")
        return
    try:
        review_doc = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        failures.append(f"gate {name} root review record is not readable JSON")
        return
    if (review_doc.get("gate") != name or review_doc.get("reviewer") != "root"
            or review_doc.get("decision") != "approved"
            or review_doc.get("evidence_sha256") != evidence_hashes):
        failures.append(f"gate {name} root review record does not attest to the current concrete evidence hashes")


def evaluate(registration_path: Path, gates_path: Path) -> tuple[list[str], str]:
    registration = load_json(registration_path)
    gates = load_json(gates_path)
    failures = validate_registration(registration)
    reg_hash = sha256(registration_path)

    registered = gates.get("registration", {})
    if registered.get("path") != registration_path.relative_to(ROOT).as_posix():
        failures.append("gate file must bind the expected preregistration path")
    if registered.get("sha256") != reg_hash:
        failures.append("gate file registration hash does not match the current preregistration")

    gate_map = gates.get("gates", {})
    if gates.get("status") != "ready":
        failures.append("gate file status must be ready after all evidence is reviewed")
    for name in REQUIRED_GATES:
        entry = gate_map.get(name)
        if not isinstance(entry, dict) or entry.get("status") != "passed":
            failures.append(f"gate {name} is not marked passed")
            continue
        _validate_gate_evidence(name, entry, failures)

    if failures:
        return failures, reg_hash
    return [], reg_hash


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, default=repo_file(DEFAULT_REGISTRATION.as_posix()))
    parser.add_argument("--gates", type=Path, default=repo_file(DEFAULT_GATES.as_posix()))
    args = parser.parse_args(argv)
    try:
        failures, reg_hash = evaluate(args.registration.resolve(), args.gates.resolve())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    if failures:
        print(f"BLOCKED. Preregistration SHA-256: {reg_hash}")
        for failure in failures:
            print(f"- {failure}")
        return 2

    print(f"READY. Preregistration SHA-256: {reg_hash}")
    print("Run baselines, then the four seed-0 trials sequentially; freeze the development winner.")
    print("Run exactly two confirmation trials for that frozen recipe with seeds 1 and 2.")
    print("This is an attestation audit only; it does not validate model, corpus, hardware, score-runtime, or method results.")
    print("Do not open the locked test until root has reviewed the actual evidence and the candidate, temperature, threshold, and interval code hash are frozen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
