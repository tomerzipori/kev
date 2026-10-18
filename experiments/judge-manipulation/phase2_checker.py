"""Blinded local checker client and deterministic disposition rules for Phase 2."""
from __future__ import annotations

import json
from pathlib import Path
import urllib.error
import urllib.request
from collections import Counter
from typing import Any

from phase2_pilot import (
    HERE, RUN_DIR, SPECIAL_MARKER_RE, PilotError, load_frozen_inputs,
    load_parent_tokenizer, exact_occurrences, read_json, sha256_bytes, sha256_file,
    validate_candidate, validate_global_response_uniqueness, validate_label_source,
    lineage_components, write_json, write_jsonl,
)

def checker_messages(candidate: dict[str, Any], prompt: dict[str, Any]) -> list[dict[str, str]]:
    """Return only frozen prompts and the response as a JSON string; no metadata crosses this boundary."""
    raw = candidate["state"]
    encoded = json.dumps(raw, ensure_ascii=False)
    # Prevent literal control-token spellings inside user data from becoming chat control tokens.
    encoded = SPECIAL_MARKER_RE.sub(lambda m: "\\u003c|" + m.group(1) + "|\\u003e", encoded)
    user = prompt["prompts"]["user_template"].replace("[[RESPONSE_JSON_STRING]]", encoded)
    return [
        {"role": "system", "content": prompt["prompts"]["system"]},
        {"role": "user", "content": user},
    ]


def _post_json(url: str, payload: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not isinstance(result, dict):
        raise PilotError("checker endpoint returned a non-object JSON response")
    return result


def _get_json(url: str, timeout: int = 20) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not isinstance(result, dict):
        raise PilotError(f"{url}: expected JSON object")
    return result


def _model_file_identity(model_config: dict[str, Any]) -> dict[str, str]:
    """Hash the exact local GGUF and tokenizer files named by the runtime lock."""
    model_path = Path(model_config["gguf_path"])
    expected_model_hash = model_config["gguf_sha256"]
    if not model_path.is_file():
        raise PilotError(f"{model_config.get('id')}: runtime lock lacks a readable pinned model path/hash")
    actual_model_hash = sha256_file(model_path)
    if actual_model_hash != expected_model_hash:
        raise PilotError(f"{model_config['id']}: GGUF/model hash mismatch: {actual_model_hash}")

    tokenizer_path = Path(model_config["tokenizer_path"])
    tokenizer_files = model_config["tokenizer_files"]
    if not tokenizer_path.is_dir() or not isinstance(tokenizer_files, list) or not tokenizer_files:
        raise PilotError(f"{model_config['id']}: runtime lock must pin a tokenizer directory and its complete file list")
    verified_tokenizer_hashes: list[tuple[str, str]] = []
    for entry in tokenizer_files:
        path = Path(entry["path"])
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            raise PilotError(f"{model_config['id']}: tokenizer file hash mismatch for {path}")
        verified_tokenizer_hashes.append((str(path), actual))
    tokenizer_digest = sha256_bytes(json.dumps(verified_tokenizer_hashes, separators=(",", ":")).encode())
    return {"gguf_sha256": actual_model_hash, "tokenizer_files_sha256": tokenizer_digest}


def verify_live_runtime_identity(base_url: str, model_id: str, runtime: dict[str, Any]) -> dict[str, Any]:
    """Bind a checker alias to the local binary/tokenizer hashes and the currently loaded server model."""
    model_config = next((model for model in runtime["models"] if model["id"] == model_id), None)
    if model_config is None:
        raise PilotError(f"{model_id}: absent from the pinned runtime lock")
    hashes = _model_file_identity(model_config)
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    props = _get_json(root + "/props")
    live_alias = props.get("model_alias")
    live_model_path = props.get("model_path")
    default_generation = props.get("default_generation_settings")
    if not isinstance(live_alias, str) or live_alias != model_id:
        raise PilotError(f"live llama-server alias {live_alias!r} does not match pinned {model_id!r}")
    if not isinstance(live_model_path, str) or not Path(live_model_path).is_absolute():
        raise PilotError("live /props lacks an absolute loaded model_path")
    expected_path = Path(model_config["gguf_path"]).resolve(strict=True)
    actual_path = Path(live_model_path).resolve(strict=True)
    if actual_path != expected_path:
        raise PilotError(f"live llama-server model path {actual_path} does not match pinned {expected_path}")
    if not isinstance(default_generation, dict):
        raise PilotError("live /props lacks default_generation_settings")
    reported_context = default_generation.get("n_ctx")
    if type(reported_context) is not int:
        raise PilotError("live /props lacks integer default_generation_settings.n_ctx")
    if reported_context != int(model_config["context_size"]):
        raise PilotError(f"live llama-server context {reported_context} differs from pinned {model_config['context_size']}")
    models = _get_json(base_url.rstrip("/") + "/models").get("data")
    if not isinstance(models, list) or model_id not in {
        entry.get("id") for entry in models if isinstance(entry, dict)
    }:
        raise PilotError(f"live /v1/models does not report expected alias {model_id!r}")
    return {
        "model_id": model_id,
        "gguf_path": str(expected_path),
        **hashes,
        "live_model_alias": live_alias,
        "live_model_path": str(actual_path),
        "live_context_size": reported_context,
        "advertised_model_ids": sorted(entry["id"] for entry in models if isinstance(entry, dict) and isinstance(entry.get("id"), str)),
    }


def _output_binding(candidate_file_hash: str, runtime_hash: str, prompt_file_hash: str, model_id: str, response_hash: str) -> dict[str, str]:
    return {
        "candidate_file_sha256": candidate_file_hash,
        "runtime_lock_sha256": runtime_hash,
        "checker_prompt_file_sha256": prompt_file_hash,
        "model_id": model_id,
        "response_sha256": response_hash,
    }


def _verify_output_binding(item: dict[str, Any], expected: dict[str, str], candidate_id: str) -> None:
    actual = {key: item.get(key) for key in expected}
    if actual != expected:
        raise PilotError(f"{candidate_id}: stale or mismatched checker output binding: {actual}")


def load_bound_output_log(
    path: Path,
    candidates: list[dict[str, Any]],
    candidate_file_hash: str,
    runtime_hash: str,
    prompt_hash: str,
    model_id: str,
) -> dict[str, dict[str, Any]]:
    """Load only attempts bound to the exact current candidate, response, prompt, runtime, and model."""
    by_id = {row["candidate_id"]: row for row in candidates}
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            item = json.loads(line)
            candidate_id = item.get("candidate_id")
            if candidate_id not in by_id:
                raise PilotError(f"stale checker log contains unknown candidate {candidate_id!r}")
            candidate = by_id[candidate_id]
            expected = _output_binding(
                candidate_file_hash,
                runtime_hash,
                prompt_hash,
                model_id,
                candidate["response_sha256"],
            )
            _verify_output_binding(item, expected, candidate_id)
            latest[candidate_id] = item
    return latest


def preflight_context(
    base_url: str,
    model_id: str,
    messages: list[dict[str, str]],
    response_format: dict[str, Any],
    runtime: dict[str, Any],
    max_tokens: int,
) -> dict[str, Any]:
    model_config = next((m for m in runtime["models"] if m["id"] == model_id), None)
    if model_config is None:
        raise PilotError(f"{model_id}: absent from frozen checker runtime lock")
    context_size = int(model_config["context_size"])
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "add_generation_prompt": True,
        "response_format": response_format,
    }
    if model_id == "qwen3-4b-q4km":
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    api_root = base_url.rstrip("/")
    if api_root.endswith("/v1"):
        api_root = api_root[:-3]
    applied = _post_json(api_root + "/apply-template", payload)
    rendered = applied.get("prompt")
    if not isinstance(rendered, str):
        raise PilotError("/apply-template did not return a prompt string")
    tokenized = _post_json(api_root + "/tokenize", {
        "content": rendered,
        "add_special": True,
        "parse_special": True,
        "with_pieces": False,
    })
    tokens = tokenized.get("tokens")
    if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
        raise PilotError("/tokenize did not return an integer token array")
    prompt_tokens = len(tokens)
    return {
        "prompt": rendered,
        "prompt_tokens": prompt_tokens,
        "context_size": context_size,
        "max_tokens": max_tokens,
        "fits": prompt_tokens + max_tokens <= context_size,
        "prompt_sha256": sha256_bytes(rendered.encode("utf-8")),
    }


def parse_checker_output(raw_output: str, response: str) -> dict[str, Any]:
    """Strictly parse the schema and derive all exact evidence offsets from the original response."""
    def no_duplicate_keys(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise PilotError(f"duplicate JSON key {key!r}")
            obj[key] = value
        return obj

    try:
        value = json.loads(raw_output, object_pairs_hook=no_duplicate_keys, parse_constant=lambda x: (_ for _ in ()).throw(PilotError(f"invalid JSON constant {x}")))
    except (json.JSONDecodeError, TypeError) as exc:
        raise PilotError(f"malformed checker JSON: {exc}") from exc
    if not isinstance(value, dict) or set(value) != {"supported", "manipulation", "evidence"}:
        raise PilotError("checker output must have exactly supported, manipulation, and evidence fields")
    if type(value["supported"]) is not bool or type(value["manipulation"]) is not bool:
        raise PilotError("supported and manipulation must be exact JSON booleans")
    if not value["supported"] and value["manipulation"] is not False:
        raise PilotError("unsupported checker output must set manipulation=false")
    evidence = value["evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise PilotError("checker evidence must be a non-empty array")
    derived = []
    for span in evidence:
        derived.append({"text": span, "occurrences": exact_occurrences(response, span)})
    return {"supported": value["supported"], "manipulation": value["manipulation"], "evidence": derived}


def decide_disposition(candidate: dict[str, Any], model_results: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    """Apply source hierarchy and two-checker agreement without overwriting construction labels."""
    if set(model_results) != {"qwen3-4b-q4km", "phi4-mini-q4km"}:
        return {"disposition": "quarantine_inference_failure", "reason": "both pinned checker outcomes are required"}
    parsed: dict[str, dict[str, Any]] = {}
    for model_id, result in model_results.items():
        if result is not None and result.get("context_fit") is False:
            return {"disposition": "unsupported_context", "reason": f"{model_id} complete prompt plus output budget exceeds context"}
        if result is None or result.get("error"):
            return {"disposition": "quarantine_inference_failure", "reason": f"{model_id} output is missing or failed"}
        try:
            parsed[model_id] = parse_checker_output(result["raw_output"], candidate["state"])
        except (PilotError, KeyError, TypeError) as exc:
            return {"disposition": "quarantine_malformed_or_unsupported", "reason": f"{model_id}: {exc}"}
    if not all(output["supported"] for output in parsed.values()):
        return {"disposition": "quarantine_malformed_or_unsupported", "reason": "one or more checkers marked the response unsupported"}
    decisions = [parsed[m]["manipulation"] for m in ("qwen3-4b-q4km", "phi4-mini-q4km")]
    if decisions[0] is not decisions[1]:
        return {"disposition": "quarantine_checker_disagreement", "reason": "valid checker decisions disagree"}
    label = decisions[0]
    if candidate["source"] == "sourcefree_construction":
        known = validate_label_source(candidate["source"], candidate.get("expected_label"))
        if label is not known:
            return {"disposition": "quarantine_checker_disagreement", "reason": "checker agreement conflicts with frozen construction label", "expected_label": known, "checker_label": label}
        disposition = "accepted_known_construction"
        accepted_label = known
    elif candidate["source"] == "oasst1":
        if candidate.get("reservation") != "calibration" or candidate.get("expected_label") is not None:
            return {"disposition": "excluded_source_mapping", "reason": "OASST must remain calibration-only and unlabeled before agreement"}
        disposition = "accepted_semantic_agreement"
        accepted_label = label
    else:
        return {"disposition": "excluded_source_mapping", "reason": "source has no target-label mapping"}
    evidence = {model_id: parsed[model_id]["evidence"] for model_id in parsed}
    return {"disposition": disposition, "accepted_label": accepted_label, "evidence": evidence}


def materialize_oasst_agreement(candidate: dict[str, Any], model_results: dict[str, dict[str, Any] | None], tokenizer: Any, row_number: int) -> dict[str, Any]:
    """Only two valid agreeing semantic outputs can become a labeled, strict Kev-format calibration record."""
    from kev.data import materialize
    from kev.model import encode, training_context

    if candidate.get("source") != "oasst1" or candidate.get("reservation") != "calibration" or candidate.get("expected_label") is not None:
        raise PilotError("only unlabeled calibration-only OASST candidates may be materialized after agreement")
    if set(model_results) != {"qwen3-4b-q4km", "phi4-mini-q4km"}:
        raise PilotError("semantic label requires outputs from both pinned checkers")
    parsed = {}
    for model_id, result in model_results.items():
        if not isinstance(result, dict) or result.get("error") or result.get("context_fit") is not True:
            raise PilotError(f"{model_id}: semantic record has no complete, context-admitted output")
        parsed[model_id] = parse_checker_output(result["raw_output"], candidate["state"])
        if not parsed[model_id]["supported"]:
            raise PilotError(f"{model_id}: unsupported semantic candidate cannot receive a label")
    labels = [parsed[model_id]["manipulation"] for model_id in ("qwen3-4b-q4km", "phi4-mini-q4km")]
    if labels[0] is not labels[1]:
        raise PilotError("semantic label requires exact boolean agreement")
    label = labels[0]
    if type(label) is not bool:
        raise PilotError("semantic agreement must yield an exact boolean label")

    request = candidate.get("api_request")
    if not isinstance(request, dict) or request.get("state") != candidate.get("state"):
        raise PilotError("candidate does not retain the validated System One API request")
    labelled_request = {
        **request,
        "questions": {
            qid: {
                **question,
                "label": label,
                "src": "judge_manipulation_oasst_agreement",
            }
            for qid, question in request["questions"].items()
        },
    }
    record = materialize(labelled_request)
    record["_meta"] = {
        "id": candidate["candidate_id"],
        "source": "oasst1",
        "group_id": candidate["group_id"],
        "row": row_number,
        "split": "calibration",
        "variant": "clean",
        "response_sha256": candidate["response_sha256"],
        "source_provenance": {
            key: candidate["source_provenance"].get(key)
            for key in ("dataset", "revision", "file", "file_sha256", "license", "message_id", "parent_id", "message_tree_id", "selection_rank", "selection_hash")
        },
        "checker_evidence": {model_id: parsed[model_id]["evidence"] for model_id in parsed},
    }
    context = training_context(2048)
    encoded = encode(tokenizer, record, max_state=context["max_state"], max_branch=context["max_branch"], strict=True)
    if encoded["state_truncated"] or encoded["state_tokens"] > context["max_state"] or len(encoded["ids"]) > context["max_packed"]:
        raise PilotError("accepted OASST record fails the strict pinned Kev training-context boundary")
    questions = record["questions"]
    if not isinstance(questions, list) or len(questions) != 1:
        raise PilotError("accepted OASST record lost the single frozen Kev question")
    question = questions[0]
    if question.get("qid") != "judge_manipulation" or type(question.get("label")) is not int or question["label"] != int(label):
        raise PilotError("accepted OASST record did not materialize the agreed boolean into Kev's integer label mapping")
    return record


def validate_accepted_semantic_loader(records: list[dict[str, Any]], directory: Path | None = None) -> str:
    """Round-trip accepted semantic labels through the real Kev suite loader in a temporary run directory."""
    from kev.suite import load_split

    directory = directory or (RUN_DIR / "accepted-semantic-validation")
    partition = directory / "calibration.jsonl"
    write_jsonl(partition, records)
    manifest = {
        "schema_version": 1,
        "name": "judge-manipulation-phase2-accepted-oasst-validation-only",
        "temporary": True,
        "not_a_frozen_training_or_evaluation_suite": True,
        "files": {"calibration.jsonl": {"sha256": sha256_file(partition), "records": len(records)}},
    }
    write_json(directory / "manifest.json", manifest)
    loaded = load_split(directory, "calibration")
    if [row.get("_meta", {}).get("id") for row in loaded] != [row["_meta"]["id"] for row in records]:
        raise PilotError("Kev suite loader changed accepted semantic record IDs or ordering")
    for row in loaded:
        questions = row.get("questions")
        if not isinstance(questions, list) or len(questions) != 1:
            raise PilotError(f"{row.get('_meta', {}).get('id')}: invalid accepted Kev question list")
        question = questions[0]
        if question.get("qid") != "judge_manipulation" or type(question.get("label")) is not int or question["label"] not in (0, 1):
            raise PilotError(f"{row.get('_meta', {}).get('id')}: accepted semantic label is not a Kev integer mapping")
        if not row.get("_meta", {}).get("checker_evidence"):
            raise PilotError(f"{row.get('_meta', {}).get('id')}: accepted semantic evidence metadata was lost")
    return sha256_file(partition)


def structural_pilot_gate(
    candidates: list[dict[str, Any]],
    dispositions: list[dict[str, Any]],
    outputs: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Require complete unique coverage and nonempty accepted examples from both label strata and OASST."""
    failures = []
    candidate_ids = [row.get("candidate_id") for row in candidates]
    disposition_ids = [row.get("candidate_id") for row in dispositions]
    if len(candidates) != 200 or len(set(candidate_ids)) != 200:
        failures.append("candidate set is not exactly 200 unique rows")
    if len(dispositions) != 200 or len(set(disposition_ids)) != 200 or set(disposition_ids) != set(candidate_ids):
        failures.append("every candidate lacks exactly one explicit disposition")

    model_counts = {}
    for model_id in ("qwen3-4b-q4km", "phi4-mini-q4km"):
        model_results = outputs.get(model_id, {})
        valid = 0
        for candidate in candidates:
            result = model_results.get(candidate.get("candidate_id"))
            if not isinstance(result, dict) or result.get("error") or result.get("context_fit") is False:
                continue
            try:
                parsed = parse_checker_output(result["raw_output"], candidate["state"])
            except (PilotError, KeyError, TypeError):
                continue
            if parsed["supported"]:
                valid += 1
        model_counts[model_id] = {"observed": len(model_results), "valid_supported": valid}
        if len(model_results) != 200:
            failures.append(f"{model_id} lacks output records for all 200 candidates")
        if valid == 0:
            failures.append(f"{model_id} has no valid supported actual output")

    accepted = [row for row in dispositions if row.get("disposition") in ("accepted_known_construction", "accepted_semantic_agreement")]
    accepted_counts = {
        "known_positive": sum(row.get("disposition") == "accepted_known_construction" and row.get("accepted_label") is True for row in accepted),
        "known_negative": sum(row.get("disposition") == "accepted_known_construction" and row.get("accepted_label") is False for row in accepted),
        "oasst_semantic_agreement": sum(row.get("disposition") == "accepted_semantic_agreement" and row.get("source") == "oasst1" for row in accepted),
    }
    for name, count in accepted_counts.items():
        if count == 0:
            failures.append(f"no accepted {name} row")
    def has_both_evidence(row: dict[str, Any]) -> bool:
        evidence = row.get("evidence")
        if not isinstance(evidence, dict) or set(evidence) != {"qwen3-4b-q4km", "phi4-mini-q4km"}:
            return False
        return all(
            isinstance(spans, list)
            and bool(spans)
            and all(
                isinstance(span, dict)
                and isinstance(span.get("text"), str)
                and bool(span["text"])
                and isinstance(span.get("occurrences"), list)
                and bool(span["occurrences"])
                for span in spans
            )
            for spans in evidence.values()
        )

    if any(not has_both_evidence(row) for row in accepted):
        failures.append("an accepted row lacks both checkers' validated evidence")
    return {
        "structural_pass": not failures,
        "structural_gate_failures": failures,
        "candidate_count": len(candidates),
        "disposition_count": len(dispositions),
        "model_output_counts": model_counts,
        "accepted_counts": accepted_counts,
    }


def load_runtime_lock() -> dict[str, Any]:
    path = HERE / "checker-runtime-lock-phase-2.json"
    if not path.is_file():
        raise PilotError("checker runtime lock has not been reviewed/materialized")
    runtime = read_json(path)
    if runtime.get("server", {}).get("base_url") not in ("http://127.0.0.1:18082/v1", "http://127.0.0.1:18082"):
        raise PilotError("runtime server URL differs from the registered local interface")
    return runtime


def load_frozen_candidates() -> tuple[list[dict[str, Any]], str, Any]:
    """Reload only the manifest-bound 200 rows and rerun strict Kev admission before checking or accepting them."""
    from kev.suite import read_jsonl

    _, prompt, _, _, parent_lock = load_frozen_inputs()
    manifest = read_json(HERE / "pilot-candidate-manifest-phase-2.json")
    candidate_path = RUN_DIR / "candidates.jsonl"
    candidate_hash = sha256_file(candidate_path)
    if candidate_hash != manifest.get("candidate_jsonl_sha256") or manifest.get("candidate_count") != 200:
        raise PilotError("candidate JSONL is not the exact 200-row manifest-bound artifact")
    frozen_hashes = {
        "construction_lock_sha256": sha256_file(HERE / "construction-template-lock-phase-2.json"),
        "checker_prompt_lock_sha256": sha256_file(HERE / "checker-prompts-phase-2.json"),
        "pilot_plan_sha256": sha256_file(HERE / "pilot-plan-phase-2.json"),
        "source_registry_sha256": sha256_file(HERE / "source-registry.json"),
        "phase4_model_lock_sha256": sha256_file(HERE / "model-lock-phase-4.json"),
        "parent_tokenizer_sha256": parent_lock["parent"]["files"]["tokenizer.json"]["sha256"],
    }
    if any(manifest.get(key) != value for key, value in frozen_hashes.items()):
        raise PilotError("candidate manifest is stale relative to a frozen input")
    for key, path in (
        ("compact_index_sha256", HERE / "pilot-candidate-index-phase-2.jsonl"),
        ("oasst_selection_lock_sha256", HERE / "oasst-selection-lock-phase-2.json"),
    ):
        if sha256_file(path) != manifest.get(key):
            raise PilotError(f"manifest-bound frozen selection artifact changed: {path}")
    candidates = read_jsonl(candidate_path)
    if len(candidates) != 200 or len({row["candidate_id"] for row in candidates}) != 200:
        raise PilotError("candidate JSONL does not contain exactly 200 unique IDs")
    validate_global_response_uniqueness(candidates)
    lineage = lineage_components([
        {"id": row["candidate_id"], "reservation": row["reservation"], "lineage_keys": row["lineage_keys"]}
        for row in candidates
    ])
    tokenizer = load_parent_tokenizer(parent_lock)
    for row in candidates:
        actual = lineage[row["candidate_id"]]
        if actual["group_id"] != row.get("group_id") or actual["component_size"] != row.get("component_size"):
            raise PilotError(f"{row['candidate_id']}: stored lineage membership differs from transitive reconstruction")
        validate_candidate(row, prompt["detector_question"], tokenizer)
    return candidates, candidate_hash, tokenizer


def run_one_model(model_id: str, retry_failures: bool = False) -> dict[str, int]:
    _, prompt, _, _, _ = load_frozen_inputs()
    runtime = load_runtime_lock()
    candidates, candidate_file_hash, _ = load_frozen_candidates()
    runtime_file_hash = sha256_file(HERE / "checker-runtime-lock-phase-2.json")
    prompt_file_hash = sha256_file(HERE / "checker-prompts-phase-2.json")
    identity = verify_live_runtime_identity(runtime["server"]["base_url"], model_id, runtime)
    output_path = RUN_DIR / "checker-outputs" / f"{model_id}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    latest = load_bound_output_log(
        output_path,
        candidates,
        candidate_file_hash,
        runtime_file_hash,
        prompt_file_hash,
        model_id,
    )
    counts = Counter()
    with output_path.open("a", encoding="utf-8", newline="\n") as stream:
        for candidate in candidates:
            candidate_id = candidate["candidate_id"]
            previous = latest.get(candidate_id)
            if previous and (not previous.get("error") or not retry_failures):
                counts["resumed"] += 1
                continue
            try:
                messages = checker_messages(candidate, prompt)
                context = preflight_context(
                    runtime["server"]["base_url"],
                    model_id,
                    messages,
                    prompt["request_parameters"]["response_format"],
                    runtime,
                    prompt["request_parameters"]["max_tokens"],
                )
                if not context["fits"]:
                    item = {"candidate_id": candidate_id, "response_sha256": candidate["response_sha256"], "model_id": model_id, "context_fit": False, "prompt_tokens": context["prompt_tokens"], "context_size": context["context_size"], "max_tokens": context["max_tokens"], "prompt_sha256": context["prompt_sha256"], "error": "unsupported_context"}
                    counts["unsupported_context"] += 1
                else:
                    completion_url = runtime["server"]["base_url"].rstrip("/")
                    if not completion_url.endswith("/v1"):
                        completion_url += "/v1"
                    completion = _post_json(completion_url + "/chat/completions", {
                        "model": model_id,
                        "messages": messages,
                        "temperature": prompt["request_parameters"]["temperature"],
                        "max_tokens": prompt["request_parameters"]["max_tokens"],
                        "response_format": prompt["request_parameters"]["response_format"],
                        "stream": False,
                    })
                    choices = completion.get("choices")
                    if not isinstance(choices, list) or len(choices) != 1:
                        raise PilotError("completion must contain exactly one choice")
                    choice = choices[0]
                    message = choice.get("message", {})
                    raw_output = message.get("content")
                    if not isinstance(raw_output, str):
                        raise PilotError("completion content is not a string")
                    finish_reason = choice.get("finish_reason")
                    item = {
                        "candidate_id": candidate_id,
                        "response_sha256": candidate["response_sha256"],
                        "model_id": model_id,
                        "context_fit": True,
                        "prompt_tokens": context["prompt_tokens"],
                        "context_size": context["context_size"],
                        "max_tokens": context["max_tokens"],
                        "prompt_sha256": context["prompt_sha256"],
                        "finish_reason": finish_reason,
                        "raw_output": raw_output,
                    }
                    if finish_reason != prompt["request_parameters"]["completion_finish_reason_required"]:
                        item["error"] = f"completion finish_reason {finish_reason!r} is not accepted"
                        counts["incomplete_or_truncated"] += 1
                    else:
                        counts["completed"] += 1
            except (urllib.error.URLError, TimeoutError, OSError, PilotError, KeyError, ValueError) as exc:
                item = {"candidate_id": candidate_id, "response_sha256": candidate["response_sha256"], "model_id": model_id, "error": f"{type(exc).__name__}: {exc}"}
                counts["inference_failure"] += 1
            item.update(_output_binding(candidate_file_hash, runtime_file_hash, prompt_file_hash, model_id, candidate["response_sha256"]))
            item["runtime_identity"] = identity
            stream.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
    return dict(counts)


def build_dispositions() -> dict[str, Any]:
    candidates, candidate_file_hash, tokenizer = load_frozen_candidates()
    runtime_file_hash = sha256_file(HERE / "checker-runtime-lock-phase-2.json")
    prompt_file_hash = sha256_file(HERE / "checker-prompts-phase-2.json")
    outputs = {}
    for model_id in ("qwen3-4b-q4km", "phi4-mini-q4km"):
        path = RUN_DIR / "checker-outputs" / f"{model_id}.jsonl"
        outputs[model_id] = load_bound_output_log(
            path,
            candidates,
            candidate_file_hash,
            runtime_file_hash,
            prompt_file_hash,
            model_id,
        )
    rows = []
    for candidate in candidates:
        result = decide_disposition(candidate, {model: outputs[model].get(candidate["candidate_id"]) for model in outputs})
        rows.append({
            "candidate_id": candidate["candidate_id"],
            "source": candidate["source"],
            "reservation": candidate["reservation"],
            "response_sha256": candidate["response_sha256"],
            "group_id": candidate["group_id"],
            **result,
        })
    accepted_semantic_records = []
    for candidate, disposition in zip(candidates, rows):
        if disposition["disposition"] == "accepted_semantic_agreement":
            model_results = {model: outputs[model][candidate["candidate_id"]] for model in outputs}
            record = materialize_oasst_agreement(candidate, model_results, tokenizer, len(accepted_semantic_records))
            if record["questions"][0]["label"] != int(disposition["accepted_label"]):
                raise PilotError(f"{candidate['candidate_id']}: accepted label changed during Kev materialization")
            accepted_semantic_records.append(record)
    semantic_rows_path = RUN_DIR / "accepted-semantic-labels.jsonl"
    write_jsonl(semantic_rows_path, accepted_semantic_records)
    semantic_records_sha256 = sha256_file(semantic_rows_path)
    semantic_loader_sha256 = validate_accepted_semantic_loader(accepted_semantic_records) if accepted_semantic_records else None
    structural = structural_pilot_gate(candidates, rows, outputs)
    write_jsonl(HERE / "pilot-dispositions-phase-2.jsonl", rows)
    report = {
        "schema_version": 1,
        "phase": 2,
        **structural,
        "disposition_counts": dict(Counter(r["disposition"] for r in rows)),
        "accepted_label_counts": dict(Counter(str(r.get("accepted_label")) for r in rows if "accepted_label" in r)),
        "accepted_semantic_kev_record_count": len(accepted_semantic_records),
        "accepted_semantic_kev_records_sha256": semantic_records_sha256,
        "accepted_semantic_loader_partition_sha256": semantic_loader_sha256,
        "dispositions_sha256": sha256_file(HERE / "pilot-dispositions-phase-2.jsonl"),
        "raw_checker_outputs_remain_ignored": True,
    }
    write_json(HERE / "pilot-disposition-report-phase-2.json", report)
    return report
