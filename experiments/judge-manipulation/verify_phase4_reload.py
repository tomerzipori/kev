"""Reload the saved training checkpoint with the same unmerged BF16 construction and compare outputs."""
import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

HF_CACHE = Path(os.environ.get("JUDGE_MANIPULATION_PHASE4_HF_HOME", str(Path.home() / ".cache/huggingface-phase4"))).expanduser().resolve()
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE / "hub")
HF_CACHE.mkdir(parents=True, exist_ok=True)

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from kev.checkpoint import Checkpoint, Meta
from kev.model import DecisionModel, load_tokenizer, training_context
from kev.data import load_records, materialize
from kev.device import allocated_bytes
from kev.suite import digest
from phase4_model_evidence import parameter_dtype_evidence


def set_stage(path, stage, **details):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"stage": stage, "updated_utc": datetime.now(timezone.utc).isoformat(), **details}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def tensors_diff(before, after):
    if set(before) != set(after):
        raise RuntimeError("saved adapter tensor names differ from the locked parent")
    changes = []
    for name in sorted(before):
        left, right = before[name].float(), after[name].float()
        if left.shape != right.shape:
            raise RuntimeError(f"adapter tensor shape changed: {name}")
        delta = right - left
        changes.append({"name": name, "max_abs_delta": float(delta.abs().max()), "l2_delta": float(delta.norm())})
    return changes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-lock", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--proof", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--stage-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.stage_file.parent.mkdir(parents=True, exist_ok=True)
    set_stage(args.stage_file, "reload_model_load")
    lock = json.loads(args.model_lock.read_text(encoding="utf-8"))
    proof = torch.load(args.proof, map_location="cpu", weights_only=False)
    parent = lock["parent"]
    base = lock["base"]
    tokenizer = load_tokenizer(base["repo"], revision=base["revision"])
    request = load_records(args.fixture)[0]
    rec = materialize(request)
    context = training_context(2048)
    checkpoint = Checkpoint(args.checkpoint)
    ours = Meta(base=base["repo"], base_revision=base["revision"], lora=lock["recipe"]["lora_rank"],
                head_dim=256, option_isolation=False, special_embeddings=False,
                weights_dtype="bf16", weights="lora")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = DecisionModel(base["repo"], tokenizer, "cuda", lora=16, revision=base["revision"],
                          head_dim=256, lora_targets="all", dtype=torch.bfloat16)
    set_stage(args.stage_file, "reload_warm_start")
    warm_start = checkpoint.warm_start(model, ours)
    set_stage(args.stage_file, "reload_warm_start_complete", warm_start_coverage=warm_start)
    model.eval()
    enc = model.encode(tokenizer, rec, strict=True, max_state=context["max_state"], max_branch=context["max_branch"])
    if enc["state_tokens"] != 2048:
        raise RuntimeError(f"reload fixture has {enc['state_tokens']} state tokens, expected 2048")
    set_stage(args.stage_file, "reload_forward")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        after = model.forward_batch([enc], shared_prefix=False)
    set_stage(args.stage_file, "reload_forward_complete")
    before_logits = torch.tensor(proof["logits"][0][0], dtype=torch.float32)
    after_logits = after[0][0].detach().float().cpu()
    max_abs_diff = float((before_logits - after_logits).abs().max())
    exact = torch.equal(before_logits, after_logits)
    if not exact:
        raise RuntimeError(f"saved/reloaded outputs differ: max absolute logit delta {max_abs_diff}")

    parent_artifacts = {}
    for filename in ("adapter_model.safetensors", "head.pt"):
        expected_sha = parent["files"][filename]["sha256"]
        resolved_path = Path(hf_hub_download(repo_id=parent["repo"], filename=filename, revision=parent["weights_revision"]))
        actual_sha = digest(resolved_path)
        if actual_sha != expected_sha:
            raise RuntimeError(f"pinned parent {filename} SHA-256 {actual_sha} does not match locked {expected_sha}")
        parent_artifacts[filename] = {"path": str(resolved_path), "sha256": actual_sha}

    adapter_path = Path(parent_artifacts["adapter_model.safetensors"]["path"])
    parent_adapter = load_file(str(adapter_path), device="cpu")
    saved_adapter_path = args.checkpoint / "adapter_model.safetensors"
    saved_adapter = load_file(str(saved_adapter_path), device="cpu")
    adapter_deltas = tensors_diff(parent_adapter, saved_adapter)
    if not any(item["max_abs_delta"] > 0 for item in adapter_deltas):
        raise RuntimeError("training did not change any LoRA adapter tensor")

    parent_checkpoint = Checkpoint(Path(parent_artifacts["head.pt"]["path"]).parent)
    parent_head = parent_checkpoint.meta.head
    saved_meta = checkpoint.meta
    head_deltas = tensors_diff(parent_head, saved_meta.head)
    if not any(item["max_abs_delta"] > 0 for item in head_deltas):
        raise RuntimeError("training did not change any pointer head tensor")

    dtype_evidence = parameter_dtype_evidence(model)
    lora_dtypes = sorted(dtype_evidence["lora_trainable_parameter_dtype_counts"])
    head_dtypes = sorted({str(param.dtype).removeprefix("torch.") for param in model.head.parameters()})
    backbone_dtypes = sorted(dtype_evidence["frozen_parameter_dtype_counts"])
    if head_dtypes != ["float32"]:
        raise RuntimeError(f"unexpected dtypes after warm start: backbone={backbone_dtypes}, LoRA={lora_dtypes}, head={head_dtypes}")

    metrics = json.loads((args.checkpoint / "training_metrics.json").read_text(encoding="utf-8"))
    if metrics.get("optimizer_steps") != 2:
        raise RuntimeError(f"expected exactly two optimizer updates, found {metrics.get('optimizer_steps')!r}")
    grad_norms = metrics.get("grad_norm", [])
    if sum(int(epoch.get("steps", 0)) for epoch in grad_norms) != 2:
        raise RuntimeError(f"training metrics do not report two gradient norms: {grad_norms}")
    if any(not math.isfinite(float(epoch.get(key, float("nan")))) or float(epoch.get(key, 0)) <= 0 for epoch in grad_norms for key in ("mean", "max")):
        raise RuntimeError(f"gradient norms must be finite and nonzero: {grad_norms}")
    set_stage(args.stage_file, "reload_verification")
    result = {
        "status": "passed",
        "checkpoint": str(args.checkpoint.resolve()),
        "parent_warm_start": warm_start,
        "parent_artifact_resolution": parent_artifacts,
        "state_tokens": proof["state_tokens"],
        "encoded_tokens": proof["encoded_tokens"],
        "branch_tokens": proof["branch_tokens"],
        "save_reload": {"exact_logits": exact, "max_abs_logit_delta": max_abs_diff},
        "dtypes": {"backbone": backbone_dtypes, "lora": lora_dtypes, "head": head_dtypes, "parameter_evidence": dtype_evidence},
        "adapter": {"sha256": digest(saved_adapter_path), "changed_tensor_count": sum(x["max_abs_delta"] > 0 for x in adapter_deltas),
                    "tensor_count": len(adapter_deltas), "max_abs_parameter_delta": max(x["max_abs_delta"] for x in adapter_deltas)},
        "head": {"sha256": digest(args.checkpoint / "head.pt"), "changed_tensor_count": sum(x["max_abs_delta"] > 0 for x in head_deltas),
                 "tensor_count": len(head_deltas), "max_abs_parameter_delta": max(x["max_abs_delta"] for x in head_deltas)},
        "training_metrics": metrics,
        "reload_peak_allocated_bytes": allocated_bytes("cuda"),
        "reload_peak_reserved_bytes": torch.cuda.max_memory_reserved("cuda"),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    set_stage(args.stage_file, "reload_complete")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
