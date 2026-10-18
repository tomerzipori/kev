"""Prepare and audit the frozen 200-row judge-manipulation data pilot.

This module deliberately separates source selection, construction, blinded checker
requests, checker parsing, and final disposition. It does not launch a model during
``prepare``; checker calls are an explicit later command.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PHASE2_CACHE = Path(os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface-phase2")))
os.environ.setdefault("HF_HUB_CACHE", str(PHASE2_CACHE / "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PHASE2_CACHE / "datasets"))

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
LOCK_PATH = HERE / "construction-template-lock-phase-2.json"
PROMPT_PATH = HERE / "checker-prompts-phase-2.json"
PLAN_PATH = HERE / "pilot-plan-phase-2.json"
SOURCE_REGISTRY_PATH = HERE / "source-registry.json"
PHASE4_LOCK_PATH = HERE / "model-lock-phase-4.json"
RUN_DIR = ROOT / "runs" / "judge-manipulation-phase2-pilot"
MAX_STATE = 2048
QUESTION_ID = "judge_manipulation"
SCORE_TEXT = "5/5"
SPECIAL_MARKER_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")
PARENT_TOKENIZER_FILES = (
    "added_tokens.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


class PilotError(ValueError):
    """A fail-closed pilot admission or parsing error."""


class LineageConflict(PilotError):
    """A connected lineage component crosses preassigned reservations."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    from kev.suite import digest

    return digest(path)


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def read_json(path: Path) -> Any:
    from kev.suite import read_json as suite_read_json

    return suite_read_json(path)


def write_json(path: Path, value: Any) -> None:
    from kev.suite import write_json as suite_write_json

    path.parent.mkdir(parents=True, exist_ok=True)
    suite_write_json(path, value)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    from kev.suite import write_jsonl as suite_write_jsonl

    path.parent.mkdir(parents=True, exist_ok=True)
    suite_write_jsonl(path, rows)


def _frozen_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _frozen_jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")


def write_once(path: Path, value: Any, jsonl: bool = False) -> None:
    """Write a reproducible artifact once; a changed reconstruction is an error, never an overwrite."""
    data = _frozen_jsonl_bytes(value) if jsonl else _frozen_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise PilotError(f"reconstruction differs from already frozen bytes: {path}")
        return
    if jsonl:
        from kev.suite import write_jsonl as suite_write_jsonl

        suite_write_jsonl(path, value)
    else:
        from kev.suite import write_json as suite_write_json

        suite_write_json(path, value)
    if path.read_bytes() != data:
        raise PilotError(f"canonical Kev writer produced unexpected bytes for {path}")


def write_reconstruction_manifest(path: Path, value: dict[str, Any]) -> None:
    """Allow command wording to evolve only when every frozen input/output field is byte-equivalent."""
    if not path.exists():
        write_once(path, value)
        return
    previous = read_json(path)
    mutable = {"deterministic_reconstruction_command", "workstation_wsl_command"}
    previous_frozen = {key: item for key, item in previous.items() if key not in mutable}
    next_frozen = {key: item for key, item in value.items() if key not in mutable}
    if previous_frozen != next_frozen:
        raise PilotError(f"frozen candidate manifest differs from reconstruction: {path}")
    if previous != value:
        write_json(path, value)
    if path.read_bytes() != _frozen_json_bytes(value):
        raise PilotError(f"canonical Kev writer produced unexpected manifest bytes for {path}")


def _parent_tokenizer_dir(parent_lock: dict[str, Any], override: Path | None = None) -> Path:
    files = parent_lock["parent"]["files"]
    if override is None:
        directory = Path(files["tokenizer.json"]["local_path"]).parent
    else:
        if not override.is_dir():
            raise PilotError(f"parent tokenizer override must be a directory: {override}")
        directory = override
    for name in PARENT_TOKENIZER_FILES:
        expected = files.get(name, {}).get("sha256")
        path = directory / name
        if not expected or not path.is_file() or sha256_file(path) != expected:
            raise PilotError(f"parent tokenizer file is missing or does not match its Phase 4 hash: {path}")
    return directory


def load_frozen_inputs(parent_tokenizer_dir: Path | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    lock = read_json(LOCK_PATH)
    prompt = read_json(PROMPT_PATH)
    plan = read_json(PLAN_PATH)
    source_registry = read_json(SOURCE_REGISTRY_PATH)
    parent_lock = read_json(PHASE4_LOCK_PATH)
    plan_hash = sha256_file(PLAN_PATH)
    if plan_hash != lock["pilot_plan_sha256"]:
        raise PilotError(f"pilot plan hash differs from frozen construction lock: {plan_hash}")
    if lock["registration_sha256"] != sha256_file(HERE / "registration.json"):
        raise PilotError("registration hash differs from frozen construction lock")
    if prompt["pilot_plan_sha256"] != plan_hash:
        raise PilotError("checker prompt lock references a different pilot plan")
    _parent_tokenizer_dir(parent_lock, parent_tokenizer_dir)
    return lock, prompt, plan, source_registry, parent_lock


def load_parent_tokenizer(parent_lock: dict[str, Any], tokenizer_dir: Path | None = None):
    """Load only the already-pinned parent tokenizer; do not resolve a moving Hub ref."""
    from transformers import AutoTokenizer

    directory = _parent_tokenizer_dir(parent_lock, tokenizer_dir)
    return AutoTokenizer.from_pretrained(str(directory), local_files_only=True)


def req_for_state(state: str, detector_question: str) -> dict[str, Any]:
    return {
        "state": state,
        "questions": {
            QUESTION_ID: {
                "type": "noul",
                "instructions": detector_question,
            }
        },
    }


def encode_candidate(state: str, detector_question: str, tokenizer, known_label: Any | None = None):
    """Validate the serving API shape and strict Kev encoder limits for a candidate."""
    from kev.api import SystemOneRequest, to_record
    from kev.data import materialize
    from kev.model import encode, training_context

    request = req_for_state(state, detector_question)
    context = training_context(MAX_STATE)
    if known_label is None:
        api_record, _ = to_record(SystemOneRequest.model_validate(request))
        encoded = encode(tokenizer, api_record, max_state=context["max_state"], max_branch=context["max_branch"], strict=True)
        if len(encoded["ids"]) > context["max_packed"]:
            raise PilotError("strict Kev packed record exceeds training_context(2048)")
    else:
        if type(known_label) is not bool:
            raise PilotError("known construction labels must be exact booleans before materialize")
        labelled = {
            **request,
            "questions": {
                QUESTION_ID: {
                    **request["questions"][QUESTION_ID],
                    "label": known_label,
                    "src": "judge_manipulation_known_construction",
                }
            },
        }
        api_record = materialize(labelled)
        encoded = encode(tokenizer, api_record, max_state=context["max_state"], max_branch=context["max_branch"], strict=True)
        if len(encoded["ids"]) > context["max_packed"]:
            raise PilotError("strict Kev packed record exceeds training_context(2048)")
    return api_record, encoded


def validate_label_source(source: str, label: Any) -> bool:
    """Reject non-bool labels and every source not explicitly mapped by this pilot."""
    if source != "sourcefree_construction":
        raise PilotError(f"source {source!r} has no known-construction target-label mapping")
    if type(label) is not bool:
        raise PilotError("known construction labels must be JSON booleans; coercion is forbidden")
    return label


def _find_root(parent: dict[str, str], item: str) -> str:
    while parent[item] != item:
        parent[item] = parent[parent[item]]
        item = parent[item]
    return item


def lineage_components(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Connect records transitively by ancestry keys and reject cross-reservation groups."""
    parent = {str(row["id"]): str(row["id"]) for row in items}
    reservation_by_id = {str(row["id"]): str(row["reservation"]) for row in items}
    key_owner: dict[str, str] = {}

    def union(left: str, right: str) -> None:
        a, b = _find_root(parent, left), _find_root(parent, right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for row in items:
        ident = str(row["id"])
        keys = row.get("lineage_keys")
        if not isinstance(keys, list) or not keys or any(not isinstance(k, str) or not k for k in keys):
            raise PilotError(f"{ident}: lineage_keys must be a non-empty array of strings")
        for key in keys:
            owner = key_owner.setdefault(key, ident)
            union(ident, owner)

    members: dict[str, list[str]] = defaultdict(list)
    for ident in parent:
        members[_find_root(parent, ident)].append(ident)
    result: dict[str, dict[str, Any]] = {}
    for ids in members.values():
        reservations = sorted({reservation_by_id[i] for i in ids})
        if len(reservations) != 1:
            raise LineageConflict(f"lineage component crosses reservations {reservations}: {sorted(ids)}")
        keys = sorted({k for row in items if str(row["id"]) in set(ids) for k in row["lineage_keys"]})
        group_hash = sha256_bytes("\n".join(keys).encode("utf-8"))[:16]
        group_id = f"p2-group-{group_hash}"
        for ident in ids:
            result[ident] = {"group_id": group_id, "component_size": len(ids), "reservation": reservations[0]}
    return result


def _tokens_before(tokenizer, text: str) -> int:
    """Count user-supplied tokens before a state substring, using Kev's escaping rule."""
    from kev.model import user_tokens

    return len(user_tokens(tokenizer, text)) + 1  # include the Kev <state> delimiter


def _safe_char_offset(text: str, raw_offset: int) -> int:
    return len(SPECIAL_MARKER_RE.sub(r"<¦\1¦>", text[:raw_offset]))


def token_position_at_char(tokenizer, text: str, raw_offset: int) -> int:
    """Return the actual state token index whose offset first overlaps a raw character offset."""
    safe = SPECIAL_MARKER_RE.sub(r"<¦\1¦>", text)
    safe_offset = _safe_char_offset(text, raw_offset)
    tokenized = tokenizer(safe, add_special_tokens=False, return_offsets_mapping=True)
    for index, (start, end) in enumerate(tokenized["offset_mapping"]):
        if end > safe_offset:
            return index + 1  # state delimiter occupies index zero
    return len(tokenized["input_ids"]) + 1


def exact_occurrences(text: str, span: str) -> list[dict[str, int]]:
    if not isinstance(span, str) or not span.strip():
        raise PilotError("evidence must be a non-empty exact string")
    result = []
    start = 0
    while True:
        pos = text.find(span, start)
        if pos < 0:
            break
        result.append({"start": pos, "end": pos + len(span)})
        start = pos + 1
    if not result:
        raise PilotError("evidence string is not present verbatim in the original response")
    return result


def validate_global_response_uniqueness(candidates: list[dict[str, Any]]) -> None:
    """Reject exact body reuse or inconsistent labels across every source and construction family."""
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        actual = sha256_bytes(row["state"].encode("utf-8"))
        if actual != row.get("response_sha256"):
            raise PilotError(f"{row.get('candidate_id')}: response SHA does not match original UTF-8 bytes")
        by_hash[actual].append(row)
    collisions = {digest: rows for digest, rows in by_hash.items() if len(rows) > 1}
    if collisions:
        sample = [
            (digest, [(row["candidate_id"], row["source"], row.get("expected_label")) for row in rows])
            for digest, rows in list(collisions.items())[:3]
        ]
        raise PilotError(f"exact response-byte duplicate or conflicting target labels across the whole pilot: {sample}")


def _next_filler(bank: list[str], seed: int, index: int) -> str:
    return bank[(seed + index) % len(bank)]


def _pad_to_token_floor(tokenizer, prefix: str, bank: list[str], seed: int, target: int) -> tuple[str, int]:
    chunks: list[str] = []
    current = prefix
    i = 0
    while _tokens_before(tokenizer, current) < target:
        sentence = _next_filler(bank, seed, i)
        chunks.append(sentence)
        current = "\n".join(chunks + ([prefix] if prefix else []))
        i += 1
        if i > 1000:
            raise PilotError("could not reach requested token placement with frozen filler bank")
    return current, i


def _build_state(tokenizer, body: str, directive: str, placement: str, filler_bank: list[str], seed: int) -> tuple[str, int, int]:
    first, sep, second = body.partition(". ")
    if not sep:
        raise PilotError("frozen base response must contain two sentences")
    # The split removes the period from the first piece; restore it exactly.
    first += "."
    if placement == "prefix":
        state, attack_char = directive + "\n" + body, 0
    elif placement == "middle":
        attack_char = len(first) + 1
        state = first + "\n" + directive + "\n" + second
    elif placement == "suffix":
        state, attack_char = body + "\n" + directive, len(body) + 1
    elif placement == "after512":
        prefix, _ = _pad_to_token_floor(tokenizer, "", filler_bank, seed, 512)
        state = prefix + "\n" + directive + "\n" + body
        attack_char = len(prefix) + 1
    elif placement == "near2048":
        prefix, _ = _pad_to_token_floor(tokenizer, "", filler_bank, seed, 1900)
        state = prefix + "\n" + directive + "\n" + body
        attack_char = len(prefix) + 1
        fill_index = 1
        while _tokens_before(tokenizer, state) < 2000:
            state += "\n" + _next_filler(filler_bank, seed, fill_index)
            fill_index += 1
            if fill_index > 100:
                raise PilotError("could not reach near-2048 state target")
        if _tokens_before(tokenizer, state) > MAX_STATE:
            raise PilotError("frozen near-2048 placement exceeds the strict Kev state limit")
    else:
        raise PilotError(f"unknown frozen placement {placement!r}")
    position = token_position_at_char(tokenizer, state, attack_char)
    state_tokens = _tokens_before(tokenizer, state)
    if placement == "after512" and position < 512:
        raise PilotError("after512 construction did not reach token 512")
    if placement == "near2048" and not (position >= 1900 and 2000 <= state_tokens <= MAX_STATE):
        raise PilotError(f"near2048 placement failed measured bounds: position={position}, state_tokens={state_tokens}")
    return state, position, state_tokens


def construct_known_rows(lock: dict[str, Any], prompt: dict[str, Any], tokenizer) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    generation = lock["generation"]
    families = {item["family_id"]: item for item in lock["families"]}
    pair_descriptors = [{"id": p["pair_id"], "reservation": p["reservation"], "lineage_keys": p["lineage_keys"]} for p in lock["pairs"]]
    group_map = lineage_components(pair_descriptors)
    candidates: list[dict[str, Any]] = []
    compact_rows: list[dict[str, Any]] = []
    for pair in lock["pairs"]:
        family = families[pair["family_id"]]
        bank = generation["reservation_banks"][pair["reservation"]]
        body_index = pair["body_index"]
        topic = bank["topics"][body_index % len(bank["topics"])]
        site = bank["sites"][(body_index // len(bank["topics"])) % len(bank["sites"])]
        body = bank["body_template"].format(topic=topic, site=site)
        attack_text = family["attack_template"].replace("[[SCORE]]", pair["score_text"])
        control_text = family["control_template"].replace("[[SCORE]]", pair["score_text"])
        if "[[SCORE]]" in attack_text or "[[SCORE]]" in control_text:
            raise PilotError(f"{pair['pair_id']}: unresolved template placeholder")
        component = group_map[pair["pair_id"]]
        lineage_keys = pair["lineage_keys"]
        for variant, directive, label in (("attack", attack_text, True), ("control", control_text, False)):
            state, placement_token, state_tokens = _build_state(tokenizer, body, directive, pair["placement"], bank["filler_sentences"], pair["seed"])
            validate_label_source("sourcefree_construction", label)
            api_record, encoded = encode_candidate(state, prompt["detector_question"], tokenizer, label)
            candidate_id = pair[f"{variant}_candidate_id"]
            expected_evidence = exact_occurrences(state, directive)
            evidence_start = expected_evidence[0]["start"]
            candidate = {
                "candidate_id": candidate_id,
                "source": "sourcefree_construction",
                "reservation": pair["reservation"],
                "expected_label": label,
                "state": state,
                "response_sha256": sha256_bytes(state.encode("utf-8")),
                "response_utf8_b64": base64.b64encode(state.encode("utf-8")).decode("ascii"),
                "lineage_keys": lineage_keys,
                "group_id": component["group_id"],
                "component_size": component["component_size"],
                "construction": {
                    "pair_id": pair["pair_id"],
                    "family_id": pair["family_id"],
                    "class": pair["class"],
                    "control_type": family["control_type"],
                    "variant": variant,
                    "placement": pair["placement"],
                    "placement_token": placement_token,
                    "state_tokens": state_tokens,
                    "evidence_text": directive,
                    "evidence_occurrences": expected_evidence,
                    "evidence_sha256": sha256_bytes(directive.encode("utf-8")),
                    "body_key": pair["body_key"],
                    "body_template_key": pair["body_template_key"],
                    "filler_template_key": pair["filler_template_key"],
                    "seed_key": pair["seed_key"],
                    "seed": pair["seed"],
                },
                "api_request": req_for_state(state, prompt["detector_question"]),
                "kev_record": api_record,
            }
            candidate["kev_record"]["_meta"] = {
                "id": candidate_id,
                "source": "judge_manipulation_phase2_pilot",
                "group_id": component["group_id"],
                "row": len(candidates),
                "split": pair["reservation"],
                "variant": "clean",
            }
            if encoded["state_tokens"] != state_tokens or encoded["state_truncated"]:
                raise PilotError(f"{candidate_id}: strict encode count differs from measured state count")
            candidates.append(candidate)
            compact_rows.append({
                "candidate_id": candidate_id,
                "source": "sourcefree_construction",
                "reservation": pair["reservation"],
                "expected_label": label,
                "response_sha256": candidate["response_sha256"],
                "group_id": candidate["group_id"],
                "construction": candidate["construction"],
            })
    return candidates, compact_rows


def _oasst_artifacts(source_registry: dict[str, Any]) -> list[dict[str, Any]]:
    entry = next(s for s in source_registry["sources"] if s["id"] == "oasst1")
    return [item for item in entry["artifacts"] if item["path"].endswith(".parquet")]


def select_oasst(lock: dict[str, Any], source_registry: dict[str, Any], tokenizer, detector_question: str, semantic_count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    source = next(s for s in source_registry["sources"] if s["id"] == "oasst1")
    revision = source["revision"]
    artifacts = _oasst_artifacts(source_registry)
    all_rows: list[dict[str, Any]] = []
    file_hashes: dict[str, str] = {}
    for artifact in artifacts:
        path = Path(hf_hub_download(
            repo_id="OpenAssistant/oasst1",
            filename=artifact["path"],
            repo_type="dataset",
            revision=revision,
            cache_dir=str(PHASE2_CACHE / "hub"),
        ))
        actual = sha256_file(path)
        if actual != artifact["sha256"]:
            raise PilotError(f"pinned OASST file hash mismatch for {artifact['path']}: {actual}")
        file_hashes[artifact["path"]] = actual
        table = pq.read_table(path)
        for row in table.to_pylist():
            row["_source_file"] = artifact["path"]
            all_rows.append(row)

    stages: dict[str, int] = {"source_rows": len(all_rows)}
    eligible = [r for r in all_rows if r.get("role") == "assistant" and r.get("lang") == "en"]
    stages["assistant_english"] = len(eligible)
    eligible = [r for r in eligible if isinstance(r.get("text"), str) and bool(r["text"].strip())]
    stages["nonempty_text"] = len(eligible)
    eligible = [r for r in eligible if isinstance(r.get("message_id"), str) and r["message_id"] and isinstance(r.get("message_tree_id"), str) and r["message_tree_id"]]
    stages["ids_present"] = len(eligible)
    eligible = [r for r in eligible if r.get("deleted") is False]
    stages["not_deleted"] = len(eligible)
    eligible = [r for r in eligible if type(r.get("review_count")) is int and r["review_count"] >= 1 and r.get("review_result") is True]
    stages["review_count_at_least_one_and_positive_review_result"] = len(eligible)

    # Stable deterministic ordering; labels never depend on any review, role, ranking, or toxicity field.
    def ordering_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
        raw = row["text"].encode("utf-8")
        body_hash = sha256_bytes(raw)
        preimage = "\0".join((revision, row["message_tree_id"], row["message_id"], body_hash)).encode("utf-8")
        return sha256_bytes(preimage), row["message_tree_id"], row["message_id"], body_hash

    eligible.sort(key=ordering_key)
    stages["review_eligible_before_dedup_or_length"] = len(eligible)
    trees_available = len({r["message_tree_id"] for r in eligible})
    body_seen: set[str] = set()
    tree_seen: set[str] = set()
    selected: list[dict[str, Any]] = []
    rejects = Counter()
    for row in eligible:
        tree_id = row["message_tree_id"]
        message_id = row["message_id"]
        raw = row["text"].encode("utf-8")
        body_hash = sha256_bytes(raw)
        if tree_id in tree_seen:
            rejects["tree_already_selected"] += 1
            continue
        if body_hash in body_seen:
            rejects["exact_body_duplicate"] += 1
            continue
        from kev.model import ContextOverflow
        try:
            _, encoded = encode_candidate(row["text"], detector_question, tokenizer)
        except ContextOverflow:
            rejects["strict_kev_state_over_2048"] += 1
            continue
        tree_seen.add(tree_id)
        body_seen.add(body_hash)
        selected.append(row)
        if len(selected) == semantic_count:
            break
    if len(selected) != semantic_count:
        raise PilotError(f"OASST selection admitted {len(selected)} rows, expected {semantic_count}")
    stages["trees_available_after_filters"] = trees_available
    stages["selected_unique_tree_unique_body_and_strictly_encoded"] = len(selected)
    stages["rejected_during_stable_selection"] = dict(sorted(rejects.items()))

    by_tree: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        if row.get("message_tree_id"):
            by_tree[str(row["message_tree_id"])].append(row)

    candidates: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    for rank, row in enumerate(selected, start=1):
        state = row["text"]  # preserve exact source Unicode; .strip() was eligibility only
        raw = state.encode("utf-8")
        response_sha = sha256_bytes(raw)
        ordering_sha = ordering_key(row)[0]
        _, encoded = encode_candidate(state, detector_question, tokenizer)
        tree_nodes = []
        for node in sorted(by_tree[row["message_tree_id"]], key=lambda item: str(item.get("message_id") or "")):
            node_text = node.get("text")
            tree_nodes.append({
                "message_id": node.get("message_id"),
                "parent_id": node.get("parent_id"),
                "role": node.get("role"),
                "lang": node.get("lang"),
                "deleted": node.get("deleted"),
                "text_sha256": sha256_bytes(node_text.encode("utf-8")) if isinstance(node_text, str) else None,
            })
        node_ids = {node["message_id"] for node in tree_nodes if node["message_id"]}
        for node in tree_nodes:
            if node["parent_id"] is not None and node["parent_id"] not in node_ids:
                raise PilotError(f"OASST tree {row['message_tree_id']} has parent link outside full tree: {node['parent_id']}")
        lineage_keys = [
            f"oasst-tree:{row['message_tree_id']}",
            f"oasst-message:{row['message_id']}",
            f"oasst-body:{response_sha}",
        ]
        candidate_id = f"p2-oasst-{row['message_id']}"
        candidates.append({
            "candidate_id": candidate_id,
            "source": "oasst1",
            "reservation": "calibration",
            "expected_label": None,
            "state": state,
            "response_sha256": response_sha,
            "response_utf8_b64": base64.b64encode(raw).decode("ascii"),
            "lineage_keys": lineage_keys,
            "group_id": "pending-oasst-tree",
            "component_size": 1,
            "source_provenance": {
                "dataset": "OpenAssistant/oasst1",
                "revision": revision,
                "file": row["_source_file"],
                "file_sha256": file_hashes[row["_source_file"]],
                "license": "Apache-2.0",
                "message_id": row["message_id"],
                "parent_id": row["parent_id"],
                "message_tree_id": row["message_tree_id"],
                "tree_nodes": tree_nodes,
                "selection_rank": rank,
                "selection_hash": ordering_sha,
                "review_count": row["review_count"],
                "review_result": row["review_result"],
                "deleted": row["deleted"],
                "label_source_fields_used": False,
            },
            "state_tokens": encoded["state_tokens"],
            "api_request": req_for_state(state, detector_question),
        })
        selections.append({
            "rank": rank,
            "candidate_id": candidate_id,
            "message_id": row["message_id"],
            "parent_id": row["parent_id"],
            "message_tree_id": row["message_tree_id"],
            "response_sha256": response_sha,
            "response_utf8_bytes": len(raw),
            "state_tokens_including_state_delimiter": encoded["state_tokens"],
            "selection_hash": ordering_sha,
            "source_file": row["_source_file"],
            "source_file_sha256": file_hashes[row["_source_file"]],
            "tree_nodes": tree_nodes,
        })

    selection_lock = {
        "schema_version": 1,
        "phase": 2,
        "status": "selected_before_checker_outputs",
        "source": "OpenAssistant/oasst1",
        "revision": revision,
        "license": "Apache-2.0",
        "filter": {
            "role": "assistant",
            "lang": "en",
            "text": "must be a non-empty UTF-8 string after whitespace-only eligibility check; original bytes are never trimmed or rewritten",
            "message_id": "non-empty string",
            "message_tree_id": "non-empty string",
            "deleted": "exact boolean false",
            "review_count": "exact integer >= 1",
            "review_result": "exact boolean true",
            "target_label": "not derived from any source field, including role/review/rank/toxicity",
            "duplicate_rule": "deduplicate exact UTF-8 response bytes by SHA-256 during stable selection",
            "tree_rule": "select at most one message per message_tree_id and retain every node ID/parent edge and text hash for each selected tree",
            "length_rule": "strict kev.model.encode with the pinned Phase 4 parent tokenizer and training_context(2048), including <state>",
        },
        "selection_algorithm": "Sort eligible rows by SHA256(UTF8(revision + NUL + message_tree_id + NUL + message_id + NUL + response_sha256)), then message_tree_id, message_id, response_sha256; greedily accept the first row per tree with a unique exact response hash that strictly encodes within 2048 state tokens until 40 rows are selected.",
        "parquet_sha256": file_hashes,
        "stage_counts": stages,
        "selected": selections,
    }
    return candidates, selection_lock


def validate_candidate(candidate: dict[str, Any], detector_question: str, tokenizer) -> dict[str, Any]:
    candidate_id = candidate["candidate_id"]
    if candidate.get("source") == "sourcefree_construction":
        validate_label_source(candidate["source"], candidate.get("expected_label"))
        api_record, encoded = encode_candidate(candidate["state"], detector_question, tokenizer, candidate["expected_label"])
    elif candidate.get("source") == "oasst1":
        if candidate.get("reservation") != "calibration" or candidate.get("expected_label") is not None:
            raise PilotError(f"{candidate_id}: OASST rows must be calibration-only with no automatic label")
        from kev.api import SystemOneRequest, to_record

        api_record, _ = to_record(SystemOneRequest.model_validate(candidate["api_request"]))
        from kev.model import encode, training_context

        context = training_context(MAX_STATE)
        encoded = encode(tokenizer, api_record, max_state=context["max_state"], max_branch=context["max_branch"], strict=True)
        if len(encoded["ids"]) > context["max_packed"]:
            raise PilotError("strict Kev packed record exceeds training_context(2048)")
    else:
        raise PilotError(f"{candidate_id}: source {candidate.get('source')!r} is excluded or has no pilot mapping")
    actual_hash = sha256_bytes(candidate["state"].encode("utf-8"))
    if actual_hash != candidate.get("response_sha256"):
        raise PilotError(f"{candidate_id}: response bytes changed after selection")
    if candidate.get("response_utf8_b64") != base64.b64encode(candidate["state"].encode("utf-8")).decode("ascii"):
        raise PilotError(f"{candidate_id}: raw UTF-8 byte payload changed")
    if encoded["state_tokens"] > MAX_STATE or encoded["state_truncated"]:
        raise PilotError(f"{candidate_id}: strict Kev state cap exceeded")
    return {"api_record": api_record, "state_tokens": encoded["state_tokens"], "encoded_tokens": len(encoded["ids"])}


def validate_with_kev_loader(candidates: list[dict[str, Any]]) -> None:
    """Exercise the suite loader's actual count/hash contract on a temporary pilot manifest."""
    from kev.suite import load_split

    known_rows = [row["kev_record"] for row in candidates if row["source"] == "sourcefree_construction"]
    if len(known_rows) != 160:
        raise PilotError(f"temporary Kev loader check expected 160 canonical known-label records, got {len(known_rows)}")
    validation_dir = RUN_DIR / "pilot-validation"
    partition = validation_dir / "development.jsonl"
    write_jsonl(partition, known_rows)
    manifest = {
        "schema_version": 1,
        "name": "judge-manipulation-phase2-temporary-validation-only",
        "temporary": True,
        "not_a_frozen_evaluation_suite": True,
        "files": {
            "development.jsonl": {
                "sha256": sha256_file(partition),
                "records": len(known_rows),
            }
        },
    }
    write_json(validation_dir / "manifest.json", manifest)
    loaded = load_split(validation_dir, "development")
    expected_ids = [row["_meta"]["id"] for row in known_rows]
    if len(loaded) != len(known_rows) or [r["_meta"]["id"] for r in loaded] != expected_ids:
        raise PilotError("Kev suite loader did not return the exact canonical known-label rows")
    for row in loaded:
        questions = row["questions"]
        if not isinstance(questions, list):
            raise PilotError(f"{row['_meta']['id']}: canonical Kev record questions must be a list")
        q = next((item for item in questions if item.get("qid") == QUESTION_ID), None)
        if q is None:
            raise PilotError(f"{row['_meta']['id']}: canonical Kev record lost the frozen question ID")
        if type(q.get("label")) is not int or q["label"] not in (0, 1):
            raise PilotError(f"{row['_meta']['id']}: canonical Kev loader row has an invalid integer noul label")


def prepare(parent_tokenizer_dir: Path | None = None) -> dict[str, Any]:
    lock, prompt, plan, source_registry, parent_lock = load_frozen_inputs(parent_tokenizer_dir)
    tokenizer = load_parent_tokenizer(parent_lock, parent_tokenizer_dir)
    known, compact_known = construct_known_rows(lock, prompt, tokenizer)
    semantic, selection_lock = select_oasst(lock, source_registry, tokenizer, prompt["detector_question"], plan["semantic_candidates"])
    candidates = known + semantic
    if len(candidates) != plan["candidate_count"] or len({row["candidate_id"] for row in candidates}) != len(candidates):
        raise PilotError("pilot candidate count or ID uniqueness gate failed")
    validate_global_response_uniqueness(candidates)
    all_lineage_items = [{"id": row["candidate_id"], "reservation": row["reservation"], "lineage_keys": row["lineage_keys"]} for row in candidates]
    components = lineage_components(all_lineage_items)
    for row in candidates:
        row["group_id"] = components[row["candidate_id"]]["group_id"]
        row["component_size"] = components[row["candidate_id"]]["component_size"]
        if row["source"] == "sourcefree_construction":
            row["kev_record"]["_meta"]["group_id"] = row["group_id"]
        validate_candidate(row, prompt["detector_question"], tokenizer)
    component_counts = Counter(info["group_id"] for info in components.values())
    reservations = Counter(row["reservation"] for row in candidates)
    sources = Counter(row["source"] for row in candidates)
    known_source_hashes = {
        "construction_lock_sha256": sha256_file(LOCK_PATH),
        "checker_prompt_lock_sha256": sha256_file(PROMPT_PATH),
        "pilot_plan_sha256": sha256_file(PLAN_PATH),
        "source_registry_sha256": sha256_file(SOURCE_REGISTRY_PATH),
        "phase4_model_lock_sha256": sha256_file(PHASE4_LOCK_PATH),
        "parent_tokenizer_sha256": parent_lock["parent"]["files"]["tokenizer.json"]["sha256"],
    }
    # These compact rows contain no response text or raw model output and are suitable for review.
    compact = compact_known + [{
        "candidate_id": row["candidate_id"],
        "source": row["source"],
        "reservation": row["reservation"],
        "expected_label": None,
        "response_sha256": row["response_sha256"],
        "group_id": row["group_id"],
        "source_provenance": {k: v for k, v in row["source_provenance"].items() if k != "tree_nodes"},
    } for row in semantic]
    candidate_path = RUN_DIR / "candidates.jsonl"
    index_path = HERE / "pilot-candidate-index-phase-2.jsonl"
    selection_path = HERE / "oasst-selection-lock-phase-2.json"
    write_once(candidate_path, candidates, jsonl=True)
    write_once(index_path, compact, jsonl=True)
    write_once(selection_path, selection_lock)
    validate_with_kev_loader(candidates)
    report = {
        "schema_version": 1,
        "phase": 2,
        "status": "candidate_and_schema_checkpoint_before_checker_outputs",
        "candidate_count": len(candidates),
        "source_counts": dict(sources),
        "reservation_counts": dict(reservations),
        "lineage_component_count": len(component_counts),
        "lineage_component_sizes": dict(sorted(Counter(component_counts.values()).items())),
        "synthetic_component_count": len({row["group_id"] for row in known}),
        "oasst_component_count": len({row["group_id"] for row in semantic}),
        "accepted_labels_assigned_to_oasst": 0,
        "frozen_inputs": known_source_hashes,
        "candidate_jsonl_sha256": sha256_file(RUN_DIR / "candidates.jsonl"),
        "candidate_jsonl_records": len(candidates),
        "candidate_index_sha256": sha256_file(HERE / "pilot-candidate-index-phase-2.jsonl"),
        "oasst_selection_lock_sha256": sha256_file(HERE / "oasst-selection-lock-phase-2.json"),
        "temporary_validation_manifest_sha256": sha256_file(RUN_DIR / "pilot-validation" / "manifest.json"),
        "state_tokens_including_delimiter": {
            "min": min(row.get("construction", {}).get("state_tokens", row.get("state_tokens", 0)) for row in candidates),
            "max": max(row.get("construction", {}).get("state_tokens", row.get("state_tokens", 0)) for row in candidates),
        },
        "template_families": len(lock["families"]),
        "attack_control_pairs": len(lock["pairs"]),
        "control_type_counts": dict(Counter(f["control_type"] for f in lock["families"])),
        "placement_counts": dict(Counter(p["placement"] for p in lock["pairs"])),
    }
    write_json(RUN_DIR / "prepare-report.json", report)
    public_manifest = {
        "schema_version": 1,
        "phase": 2,
        "status": "candidate_rows_frozen_before_checker_outputs",
        "deterministic_reconstruction_command": "python experiments/judge-manipulation/phase2_pilot.py prepare --parent-tokenizer <directory-containing-pinned-tokenizer.json>",
        "workstation_wsl_command": "wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd /mnt/c/Users/Tomer/Documents/GitHub/kev && .venv/bin/python experiments/judge-manipulation/phase2_pilot.py prepare --parent-tokenizer /home/tomer/.cache/huggingface-phase4/hub/models--jaredpalmer--kev-4b/snapshots/139fdd94f1b6a6ad80cc15e08fcb99cac885a101'",
        "candidate_count": len(candidates),
        "source_counts": dict(sources),
        "reservation_counts": dict(reservations),
        "candidate_jsonl_ignored_path": "runs/judge-manipulation-phase2-pilot/candidates.jsonl",
        "candidate_jsonl_sha256": report["candidate_jsonl_sha256"],
        "candidate_jsonl_records": len(candidates),
        "compact_index_path": "experiments/judge-manipulation/pilot-candidate-index-phase-2.jsonl",
        "compact_index_sha256": report["candidate_index_sha256"],
        "oasst_selection_lock_path": "experiments/judge-manipulation/oasst-selection-lock-phase-2.json",
        "oasst_selection_lock_sha256": report["oasst_selection_lock_sha256"],
        "construction_lock_sha256": known_source_hashes["construction_lock_sha256"],
        "checker_prompt_lock_sha256": known_source_hashes["checker_prompt_lock_sha256"],
        "pilot_plan_sha256": known_source_hashes["pilot_plan_sha256"],
        "source_registry_sha256": known_source_hashes["source_registry_sha256"],
        "phase4_model_lock_sha256": known_source_hashes["phase4_model_lock_sha256"],
        "parent_tokenizer_sha256": known_source_hashes["parent_tokenizer_sha256"],
        "lineage_components": report["lineage_component_count"],
        "synthetic_lineage_components": report["synthetic_component_count"],
        "oasst_tree_components": report["oasst_component_count"],
        "final_evaluation_suite": False,
    }
    write_reconstruction_manifest(HERE / "pilot-candidate-manifest-phase-2.json", public_manifest)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare", help="select and validate the 200 rows without contacting checkers")
    prepare_parser.add_argument(
        "--parent-tokenizer",
        type=Path,
        help="directory containing the pinned parent tokenizer.json (its Phase 4 SHA-256 is verified)",
    )
    run = sub.add_parser("run-model", help="sequentially check every admitted row with one reviewed local model")
    run.add_argument("--model", required=True, choices=("qwen3-4b-q4km", "phi4-mini-q4km"))
    run.add_argument("--retry-failures", action="store_true")
    sub.add_parser("dispositions", help="derive compact dispositions from both ignored model-output logs")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        print(json.dumps(prepare(args.parent_tokenizer), ensure_ascii=False, indent=2))
    elif args.command == "run-model":
        from phase2_checker import run_one_model
        print(json.dumps(run_one_model(args.model, args.retry_failures), ensure_ascii=False, indent=2))
    else:
        from phase2_checker import build_dispositions
        print(json.dumps(build_dispositions(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
