"""Version 2 disposition policy for the judge-manipulation pilot.

Known constructions are admitted from exact deterministic reconstruction and
evidence validation. Model outputs are retained as QC only. Unlabeled OASST
records keep the v1 two-checker semantic agreement rule.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import phase2_checker as v1
import phase2_pilot as pilot


HERE = Path(__file__).resolve().parent
RUN_DIR = pilot.RUN_DIR
REPORT_DIR = HERE / "pilot-reports" / "2026-10-18"
MODEL_IDS = ("qwen3-4b-q4km", "phi4-mini-q4km")
FROZEN_V1 = {
    "candidate_sha256": "bcb35908610e50c3e2cc1c787d1af1118735bcd67b9373af17ed595c1252939f",
    "qwen_ledger_sha256": "d7dd33ba85723d2f49af7a0760802cc2c921273d0c8b34e06ec6cfa220207a25",
    "phi_ledger_sha256": "79e1c4a8836050f9a404d8b408d5aa13d2d2ae698f7a4c82e68dc14a77c7456e",
    "v1_dispositions_sha256": "c3b924e4e7d6e9b8da54af40f6ba8817776e7d28524d4895a0ad98fc479c9f0d",
}


def validate_reconstructed_construction(
    candidate: dict[str, Any], reconstructed: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return a proof only when the candidate equals its frozen deterministic rebuild."""
    if candidate.get("source") != "sourcefree_construction":
        raise pilot.PilotError("construction proof requested for a non-construction source")
    if not isinstance(reconstructed, dict):
        raise pilot.PilotError("deterministic reconstruction is missing")

    fields = (
        "candidate_id", "source", "reservation", "expected_label", "state",
        "response_sha256", "lineage_keys", "group_id",
        "construction", "api_request", "kev_record",
    )
    mismatches = [field for field in fields if candidate.get(field) != reconstructed.get(field)]
    if mismatches:
        raise pilot.PilotError("candidate differs from deterministic reconstruction: " + ", ".join(mismatches))
    # component_size is a row count in the final candidate set, while the construction
    # lock's lineage map is pair-level. load_frozen_candidates independently rebuilds
    # and validates the stored row-level size together with the group ID and keys.

    construction = candidate.get("construction")
    if not isinstance(construction, dict):
        raise pilot.PilotError("candidate lacks construction evidence metadata")
    variant = construction.get("variant")
    expected_label = {"attack": True, "control": False}.get(variant)
    if expected_label is None or type(candidate.get("expected_label")) is not bool:
        raise pilot.PilotError("construction variant has no exact boolean label mapping")
    pilot.validate_label_source(candidate["source"], candidate["expected_label"])
    if candidate["expected_label"] is not expected_label:
        raise pilot.PilotError("construction variant conflicts with its deterministic label")

    evidence_text = construction.get("evidence_text")
    if not isinstance(evidence_text, str) or not evidence_text:
        raise pilot.PilotError("construction evidence text is missing")
    occurrences = pilot.exact_occurrences(candidate["state"], evidence_text)
    if not occurrences or construction.get("evidence_occurrences") != occurrences:
        raise pilot.PilotError("construction evidence is absent or its offsets do not match")
    if pilot.sha256_bytes(evidence_text.encode("utf-8")) != construction.get("evidence_sha256"):
        raise pilot.PilotError("construction evidence hash mismatch")
    if pilot.sha256_bytes(candidate["state"].encode("utf-8")) != candidate.get("response_sha256"):
        raise pilot.PilotError("construction response hash mismatch")
    return {
        "status": "verified",
        "candidate_id": candidate["candidate_id"],
        "source_label": expected_label,
        "evidence_sha256": construction["evidence_sha256"],
        "evidence_occurrence_count": len(occurrences),
    }


def _checker_qc(result: dict[str, Any] | None, state: str) -> dict[str, Any]:
    """Summarize one checker outcome without making it authoritative for known labels."""
    if not isinstance(result, dict):
        return {"status": "missing"}
    if result.get("context_fit") is False or result.get("error") == "unsupported_context":
        return {"status": "unsupported_context"}
    if result.get("error"):
        error = str(result.get("error", ""))
        status = "truncated_or_incomplete" if "finish_reason" in error else "inference_failure"
        return {"status": status}
    try:
        parsed = v1.parse_checker_output(result["raw_output"], state)
    except (pilot.PilotError, KeyError, TypeError):
        return {"status": "malformed"}
    if not parsed["supported"]:
        return {"status": "valid_unsupported"}
    return {"status": "valid_supported", "label": parsed["manipulation"]}


def _construction_disposition(
    candidate: dict[str, Any],
    model_results: dict[str, dict[str, Any] | None],
    proof: dict[str, Any],
) -> dict[str, Any]:
    qc = {model: _checker_qc(model_results.get(model), candidate["state"]) for model in MODEL_IDS}
    usable_labels = [qc[model].get("label") for model in MODEL_IDS if qc[model]["status"] == "valid_supported"]
    qc_disagreement = len(set(usable_labels)) > 1
    return {
        "disposition": "accepted_known_construction",
        "accepted_label": proof["source_label"],
        "label_authority": "deterministic_construction",
        "construction_proof": proof,
        "checker_qc": qc,
        "checker_qc_disagreement": qc_disagreement,
    }


def decide_disposition_v2(
    candidate: dict[str, Any],
    model_results: dict[str, dict[str, Any] | None],
    reconstructed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Use provenance-backed construction labels; preserve semantic agreement for OASST."""
    if candidate.get("source") == "sourcefree_construction":
        try:
            proof = validate_reconstructed_construction(candidate, reconstructed)
        except (pilot.PilotError, KeyError, TypeError) as exc:
            return {"disposition": "quarantine_invalid_construction", "reason": str(exc)}
        return _construction_disposition(candidate, model_results, proof)
    if candidate.get("source") == "oasst1":
        return v1.decide_disposition(candidate, model_results)
    return {"disposition": "excluded_source_mapping", "reason": "source has no v2 target-label rule"}


def _aggregate_qc(dispositions: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, Counter] = {model: Counter() for model in MODEL_IDS}
    paired_disagreements = 0
    for disposition in dispositions:
        qc = disposition.get("checker_qc", {})
        for model in MODEL_IDS:
            counts[model][qc.get(model, {}).get("status", "not_applicable")] += 1
        paired_disagreements += int(disposition.get("checker_qc_disagreement", False))
    return {
        "by_model": {model: dict(counts[model]) for model in MODEL_IDS},
        "construction_checker_disagreement_rows": paired_disagreements,
    }


def _group_summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for row in candidates:
        group = groups.setdefault(row["group_id"], {"source": row["source"], "reservation": row["reservation"], "rows": 0})
        group["rows"] += 1
    by_source_reservation: dict[str, dict[str, int]] = defaultdict(dict)
    for group in groups.values():
        key = f"{group['source']}:{group['reservation']}"
        by_source_reservation[key]["groups"] = by_source_reservation[key].get("groups", 0) + 1
        by_source_reservation[key]["rows"] = by_source_reservation[key].get("rows", 0) + group["rows"]
    return {"effective_group_count": len(groups), "by_source_reservation": dict(by_source_reservation)}


def _construction_design(lock: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = lock["pairs"]
    families = {item["family_id"]: item for item in lock["families"]}
    def tally(values: list[str]) -> dict[str, int]:
        return dict(sorted(Counter(values).items()))
    body_recipe_counts = Counter(row["body_template_key"] for row in pairs)
    filler_recipe_counts = Counter(row["filler_template_key"] for row in pairs)
    synthetic = [row for row in candidates if row["source"] == "sourcefree_construction"]
    groups_by_reservation: dict[str, set[str]] = defaultdict(set)
    rows_by_reservation: Counter = Counter()
    for row in synthetic:
        groups_by_reservation[row["reservation"]].add(row["group_id"])
        rows_by_reservation[row["reservation"]] += 1
    return {
        "synthetic_pair_count": len(pairs),
        "synthetic_row_count": len(synthetic),
        "pairs_by_reservation": tally([row["reservation"] for row in pairs]),
        "attack_classes_by_pair": tally([row["class"] for row in pairs]),
        "placements_by_pair": tally([row["placement"] for row in pairs]),
        "negative_control_types_by_pair": tally([families[row["family_id"]]["control_type"] for row in pairs]),
        "template_family_count": len({row["family_id"] for row in pairs}),
        "unique_body_keys": len({row["body_key"] for row in pairs}),
        "unique_seed_keys": len({row["seed_key"] for row in pairs}),
        "body_recipe_count_and_pair_counts": {
            "recipe_count": len(body_recipe_counts),
            "pair_counts": sorted(body_recipe_counts.values()),
        },
        "filler_recipe_count_and_pair_counts": {
            "recipe_count": len(filler_recipe_counts),
            "pair_counts": sorted(filler_recipe_counts.values()),
        },
        "effective_synthetic_groups_by_reservation": {
            name: {"groups": len(groups_by_reservation[name]), "rows": rows_by_reservation[name]}
            for name in sorted(groups_by_reservation)
        },
    }


def _assert_v1_bindings(candidate_hash: str, output_paths: dict[str, Path]) -> None:
    if candidate_hash != FROZEN_V1["candidate_sha256"]:
        raise pilot.PilotError("retrospective candidate bytes do not match frozen v1")
    for model, path in output_paths.items():
        expected_key = "qwen_ledger_sha256" if model == MODEL_IDS[0] else "phi_ledger_sha256"
        if pilot.sha256_file(path) != FROZEN_V1[expected_key]:
            raise pilot.PilotError(f"{model}: retrospective ledger bytes do not match frozen v1")
    disposition_path = HERE / "pilot-dispositions-phase-2.jsonl"
    if pilot.sha256_file(disposition_path) != FROZEN_V1["v1_dispositions_sha256"]:
        raise pilot.PilotError("v1 dispositions differ from their frozen hash")


def run_retrospective_v2() -> dict[str, Any]:
    """Apply v2 source authority to exact v1 rows and outputs; write aggregate-only report."""
    lock, prompt, _plan, _registry, _parent_lock = pilot.load_frozen_inputs()
    candidates, candidate_hash, tokenizer = v1.load_frozen_candidates()
    output_paths = {
        model: RUN_DIR / "checker-outputs" / f"{model}.jsonl"
        for model in MODEL_IDS
    }
    _assert_v1_bindings(candidate_hash, output_paths)

    reconstructed, _compact = pilot.construct_known_rows(lock, prompt, tokenizer)
    reconstructed_by_id = {row["candidate_id"]: row for row in reconstructed}
    if len(reconstructed_by_id) != 160:
        raise pilot.PilotError(f"expected 160 deterministic known rows, got {len(reconstructed_by_id)}")

    runtime_hash = pilot.sha256_file(HERE / "checker-runtime-lock-phase-2.json")
    prompt_hash = pilot.sha256_file(HERE / "checker-prompts-phase-2.json")
    recovery_dir = pilot.ROOT / "runs" / "judge-manipulation-phase2-runtime" / "20261018"
    external_audit_path = REPORT_DIR / "phase2-external-lineage-audit.json"
    external_audit = json.loads(external_audit_path.read_text(encoding="utf-8"))
    school_groups = external_audit["school_of_reward_hacks"]["conservative_grouping"]
    outputs = {
        model: v1.load_bound_output_log(
            output_paths[model], candidates, candidate_hash, runtime_hash, prompt_hash, model,
        )
        for model in MODEL_IDS
    }
    candidate_ids = {row["candidate_id"] for row in candidates}
    if any(set(outputs[model]) != candidate_ids for model in MODEL_IDS):
        raise pilot.PilotError("retrospective v2 requires exactly one bound v1 output per candidate and model")

    dispositions = []
    known_counts = Counter()
    semantic_counts = Counter()
    for candidate in candidates:
        model_results = {model: outputs[model][candidate["candidate_id"]] for model in MODEL_IDS}
        if candidate["source"] == "sourcefree_construction":
            result = decide_disposition_v2(candidate, model_results, reconstructed_by_id.get(candidate["candidate_id"]))
            if result.get("disposition") == "accepted_known_construction":
                known_counts[str(result["accepted_label"]).lower()] += 1
        else:
            result = decide_disposition_v2(candidate, model_results)
            semantic_counts[result["disposition"]] += 1
        dispositions.append({
            "candidate_id": candidate["candidate_id"],
            "source": candidate["source"],
            "reservation": candidate["reservation"],
            "group_id": candidate["group_id"],
            **result,
        })

    all_counts = Counter(row["disposition"] for row in dispositions)
    accepted_semantic = []
    for candidate, disposition in zip(candidates, dispositions):
        if disposition["disposition"] == "accepted_semantic_agreement":
            accepted_semantic.append(v1.materialize_oasst_agreement(
                candidate,
                {model: outputs[model][candidate["candidate_id"]] for model in MODEL_IDS},
                tokenizer,
                len(accepted_semantic),
            ))
    loader_hash = None
    if accepted_semantic:
        loader_hash = v1.validate_accepted_semantic_loader(
            accepted_semantic,
            RUN_DIR / "v2-retrospective" / "semantic-loader-validation",
        )

    qc_rows = [row for row in dispositions if row["source"] == "sourcefree_construction"]
    report = {
        "schema_version": 1,
        "protocol": "judge-manipulation-phase-2-v2-retrospective",
        "report_type": "aggregate_only_retrospective_reclassification",
        "created_on": "2026-10-18",
        "fresh_v2_pilot": False,
        "pilot_gate_pass": False,
        "pilot_gate_status": "not evaluated; v1 candidates and outputs are reused only for retrospective policy validation",
        "v1_dispositions_unchanged": True,
        "v2_semantics": {
            "construction_label_authority": "exact deterministic reconstruction from frozen construction lock plus exact evidence occurrence validation",
            "construction_checker_outputs": "QC only; cannot overwrite or veto a verified construction label",
            "oasst_label_authority": "two complete valid supported local outputs with exact boolean agreement",
            "checker_prompt_or_runtime_changed": False,
        },
        "inputs": {
            "master_plan_sha256": pilot.sha256_file(HERE.parents[1] / "MASTER-PLAN.md"),
            "construction_lock_sha256": pilot.sha256_file(HERE / "construction-template-lock-phase-2.json"),
            "pilot_plan_sha256": pilot.sha256_file(HERE / "pilot-plan-phase-2.json"),
            "source_registry_sha256": pilot.sha256_file(HERE / "source-registry.json"),
            "checker_prompt_sha256": prompt_hash,
            "runtime_lock_sha256": runtime_hash,
            "candidate_sha256": candidate_hash,
            "qwen_ledger_sha256": pilot.sha256_file(output_paths[MODEL_IDS[0]]),
            "phi_ledger_sha256": pilot.sha256_file(output_paths[MODEL_IDS[1]]),
            "v1_dispositions_sha256": pilot.sha256_file(HERE / "pilot-dispositions-phase-2.jsonl"),
            "v1_aggregate_report_sha256": pilot.sha256_file(HERE / "pilot-disposition-report-phase-2.json"),
            "bound_record_counts": {model: len(outputs[model]) for model in MODEL_IDS},
        },
        "runtime_smoke_evidence": {
            "scope": "Generic one-request smoke only; no pilot prompts or checker calls. These proofs establish the CUDA path/offload, not full pilot runtime readiness.",
            "qwen3-4b-q4km": {
                "gpu_layers_offloaded": 37,
                "gpu_layers_total": 37,
                "status": "GPU computation and smoke assertions reached; final report has a live-log read I/O gap and no raw completion body was persisted.",
                "proof_path": "experiments/judge-manipulation/qwen-recovery-proof-phase-2.json",
                "proof_sha256": pilot.sha256_file(HERE / "qwen-recovery-proof-phase-2.json"),
                "server_log_path": "runs/judge-manipulation-phase2-runtime/20261018/qwen-recovery-server.log",
                "server_log_sha256": pilot.sha256_file(recovery_dir / "qwen-recovery-server.log"),
            },
            "phi4-mini-q4km": {
                "gpu_layers_offloaded": 33,
                "gpu_layers_total": 33,
                "status": "Passed generic smoke with CUDA offload proof.",
                "smoke_path": "runs/judge-manipulation-phase2-runtime/20261018/phi-recovery-smoke.json",
                "smoke_sha256": pilot.sha256_file(recovery_dir / "phi-recovery-smoke.json"),
                "server_log_path": "runs/judge-manipulation-phase2-runtime/20261018/phi-recovery-server.log",
                "server_log_sha256": pilot.sha256_file(recovery_dir / "phi-recovery-server.log"),
            },
            "attempts_ledger_sha256": pilot.sha256_file(HERE / "runtime-attempts-phase-2.json"),
            "gpu_lease_released": True,
        },
        "counts": {
            "candidate_count": len(candidates),
            "deterministically_reconstructed_known_count": len(reconstructed_by_id),
            "accepted_known_construction": {"negative": known_counts["false"], "positive": known_counts["true"]},
            "accepted_semantic_agreement": all_counts["accepted_semantic_agreement"],
            "dispositions": dict(all_counts),
            "accepted_total": all_counts["accepted_known_construction"] + all_counts["accepted_semantic_agreement"],
        },
        "construction_checker_qc": _aggregate_qc(qc_rows),
        "construction_design_coverage": _construction_design(lock, candidates),
        "accepted_oasst_loader_partition_sha256": loader_hash,
        "external_lineage_audit": {
            "path": "experiments/judge-manipulation/pilot-reports/2026-10-18/phase2-external-lineage-audit.json",
            "sha256": pilot.sha256_file(external_audit_path),
            "school_pair_only_upper_bound": school_groups["pair_only_upper_bound"],
            "school_suffix_linked_components": school_groups["transitive_component_count"],
            "school_component_size_histogram": school_groups["transitive_component_size_histogram"],
            "school_target_eligibility_and_true_independence": "unresolved",
            "notinject_unique_prompt_groups": external_audit["notinject"]["unique_registered_group_keys"],
            "notinject_eligible_response_capacity": external_audit["locked_test_capacity"]["notinject_eligible_response_capacity"],
        },
        "lineage_coverage": _group_summary(candidates),
        "limitations": [
            "This retrospective result does not satisfy a fresh v2 pilot gate or authorize full-corpus processing, training, or Phase 3 freeze.",
            "The v1 candidate set has only three synthetic lineage groups; the v2 label rule does not create independent coverage.",
            "All original v1 dispositions and raw ledgers remain unchanged and bound to their original files.",
            "School of Reward Hacks lineage audit conservatively merges pairs sharing repeated exact 64-character response suffix hashes (transitive closure): 98 components from a pair-only upper bound of 105; target eligibility and independence remain unresolved.",
        ],
    }
    report_path = REPORT_DIR / "phase2-v2-retrospective-report.json"
    pilot.write_json(report_path, report)
    return report


if __name__ == "__main__":
    result = run_retrospective_v2()
    print(json.dumps(result, ensure_ascii=False, indent=2))
