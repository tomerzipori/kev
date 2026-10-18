"""Verify fresh-process unmerged reload parity for the amended Phase 4 run."""

import argparse
import json
import math
import time
from pathlib import Path

from phase4_shared_memory_support import CudaObservation, atomic_json, set_stage


def _eval_state(model):
    return {"model_training": model.training, "backbone_training": model.lm.training,
            "head_training": model.head.training}


def _lora_ranks(model):
    return {name: int(config.r) for name, config in getattr(model.lm, "peft_config", {}).items()}


def _tensor_changes(before, after):
    if set(before) != set(after):
        raise RuntimeError("checkpoint tensor names differ from the canonical parent")
    changed = 0
    max_delta = 0.0
    for name in sorted(before):
        left, right = before[name].float(), after[name].float()
        if left.shape != right.shape:
            raise RuntimeError(f"checkpoint tensor shape changed: {name}")
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise RuntimeError(f"checkpoint tensor contains non-finite values: {name}")
        delta = right - left
        maximum = float(delta.abs().max())
        if not math.isfinite(maximum):
            raise RuntimeError(f"checkpoint tensor delta is non-finite: {name}")
        changed += maximum > 0
        max_delta = max(max_delta, maximum)
    return {"changed_tensor_count": changed, "tensor_count": len(before), "max_abs_parameter_delta": max_delta}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-lock", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--proof", required=True, type=Path)
    parser.add_argument("--training-timing", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--stage-file", required=True, type=Path)
    parser.add_argument("--observation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite reload output: {args.output}")
    args.stage_file.parent.mkdir(parents=True, exist_ok=True)
    observer = CudaObservation(args.observation, "reload")
    observer.start()
    started = time.monotonic()
    result = {"status": "failed", "started_monotonic_ns": time.monotonic_ns()}
    try:
        set_stage(args.stage_file, "reload_pre_cuda_import", cuda_active=False)
        import torch
        from kev.checkpoint import Checkpoint, Meta
        from kev.data import load_records, materialize
        from kev.device import allocated_bytes
        from kev.model import DecisionModel, load_tokenizer, training_context
        from kev.predictors import kernel_environment
        from kev.suite import digest
        from phase4_model_evidence import parameter_dtype_evidence

        globals()["torch"] = torch
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download

        set_stage(args.stage_file, "reload_model_load", cuda_active=False)
        lock = json.loads(args.model_lock.read_text(encoding="utf-8"))
        timing = json.loads(args.training_timing.read_text(encoding="utf-8"))
        training_proof = json.loads((args.proof.parent / "training-model.json").read_text(encoding="utf-8"))
        before = torch.load(args.proof, map_location="cpu", weights_only=False)
        checkpoint = Checkpoint(args.checkpoint)
        base, parent = lock["base"], lock["parent"]
        tokenizer_started = time.monotonic()
        tokenizer = load_tokenizer(base["repo"], revision=base["revision"])
        tokenizer_load_seconds = round(time.monotonic() - tokenizer_started, 6)
        request = load_records(args.fixture)[0]
        record = materialize(request)
        context = training_context(2048)
        ours = Meta(base=base["repo"], base_revision=base["revision"], lora=lock["recipe"]["lora_rank"],
                    head_dim=256, option_isolation=False, special_embeddings=False,
                    weights_dtype="bf16", weights="lora")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        model_started = time.monotonic()
        model = DecisionModel(base["repo"], tokenizer, "cuda", lora=16, revision=base["revision"],
                              head_dim=256, lora_targets="all", dtype=torch.bfloat16)
        model_load_seconds = round(time.monotonic() - model_started, 6)
        set_stage(args.stage_file, "reload_warm_start", cuda_active=torch.cuda.is_initialized())
        warm_started = time.monotonic()
        warm_start = checkpoint.warm_start(model, ours)
        warm_start_seconds = round(time.monotonic() - warm_started, 6)
        model.head.temperature = checkpoint.meta.temperature
        model.eval()
        enc = model.encode(tokenizer, record, strict=True, max_state=context["max_state"], max_branch=context["max_branch"])
        shape = (enc["state_tokens"], len(enc["ids"]) - enc["state_tokens"], len(enc["ids"]))
        if shape != (2048, 51, 2099):
            raise RuntimeError(f"reload row shape differs from the registration: {shape}")
        forward_started = time.monotonic()
        set_stage(args.stage_file, "reload_forward", cuda_active=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model.forward_batch([enc], shared_prefix=False)
        torch.cuda.synchronize()
        reload_forward_seconds = round(time.monotonic() - forward_started, 6)
        after_logits = logits[0][0].detach().float().cpu()
        before_logits = torch.tensor(before["logits"][0][0], dtype=torch.float32)
        if not torch.isfinite(before_logits).all() or not torch.isfinite(after_logits).all():
            raise RuntimeError("pre-save or reloaded logits contain non-finite values")
        max_abs_delta = float((before_logits - after_logits).abs().max())
        if not torch.equal(before_logits, after_logits):
            raise RuntimeError(f"fresh unmerged reload logits differ; max absolute delta {max_abs_delta}")

        set_stage(args.stage_file, "reload_parent_tensor_check", cuda_active=True)
        parent_artifacts = {}
        for filename in ("adapter_model.safetensors", "head.pt"):
            expected_sha = parent["files"][filename]["sha256"]
            resolved = Path(hf_hub_download(repo_id=parent["repo"], filename=filename,
                                            revision=parent["weights_revision"], local_files_only=True))
            actual_sha = digest(resolved)
            if actual_sha != expected_sha:
                raise RuntimeError(f"locked parent {filename} hash mismatch: {actual_sha}")
            parent_artifacts[filename] = {"path": str(resolved), "sha256": actual_sha}
        parent_adapter = load_file(parent_artifacts["adapter_model.safetensors"]["path"], device="cpu")
        saved_adapter_path = args.checkpoint / "adapter_model.safetensors"
        saved_adapter = load_file(str(saved_adapter_path), device="cpu")
        adapter_changes = _tensor_changes(parent_adapter, saved_adapter)
        if not adapter_changes["changed_tensor_count"]:
            raise RuntimeError("training did not change any parent LoRA tensor")
        parent_head = Checkpoint(Path(parent_artifacts["head.pt"]["path"]).parent).meta.head
        head_changes = _tensor_changes(parent_head, checkpoint.meta.head)
        if not head_changes["changed_tensor_count"]:
            raise RuntimeError("training did not change any parent pointer-head tensor")

        eval_state = _eval_state(model)
        environment = kernel_environment(model, "cuda")
        temperature = float(model.head.temperature)
        if temperature != float(checkpoint.meta.temperature):
            raise RuntimeError("fresh model head temperature differs from the checkpoint metadata")
        dtypes = parameter_dtype_evidence(model)
        head_dtypes = sorted({str(parameter.dtype).removeprefix("torch.") for parameter in model.head.parameters()})
        if head_dtypes != ["float32"]:
            raise RuntimeError(f"reload pointer head is not FP32: {head_dtypes}")
        if sorted(dtypes["lora_trainable_parameter_dtype_counts"]) != ["float32"]:
            raise RuntimeError("reload LoRA parameters are not FP32")
        ranks = _lora_ranks(model)
        if ranks != {"default": 16}:
            raise RuntimeError(f"reload adapter is not the registered unmerged rank-16 LoRA: {ranks}")
        metrics = json.loads((args.checkpoint / "training_metrics.json").read_text(encoding="utf-8"))
        per_update_norms = timing.get("per_update_gradient_norms")
        if not isinstance(per_update_norms, list) or len(per_update_norms) != 2 or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
                for value in per_update_norms):
            raise RuntimeError(f"each optimizer update needs a finite nonzero gradient norm: {per_update_norms!r}")
        if timing.get("optimizer_updates") != 2 or timing.get("microbatches") != 32:
            raise RuntimeError("training did not complete exactly 32 microbatches and two optimizer updates")

        observer.capture(torch, allocated_bytes)
        result.update({
            "status": "passed",
            "checkpoint": str(args.checkpoint.resolve()),
            "parent_warm_start": warm_start,
            "parent_artifact_resolution": parent_artifacts,
            "shape": {"state_tokens": shape[0], "branch_tokens": shape[1], "full_encoded_row_tokens": shape[2]},
            "save_reload": {"exact_logits": True, "max_abs_logit_delta": max_abs_delta},
            "temperature_pre_save": before["temperature"],
            "temperature_meta_pre_save": before["meta_temperature"],
            "temperature_reload": temperature,
            "eval_state_pre_save": before["eval_state"],
            "eval_state_reload": eval_state,
            "tf32_pre_save": before["tf32"],
            "tf32_reload": {"matmul": torch.backends.cuda.matmul.allow_tf32,
                            "cudnn": torch.backends.cudnn.allow_tf32},
            "kernel_environment_pre_save": training_proof["kernel_environment"],
            "kernel_environment_reload": environment,
            "unmerged_lora": True,
            "adapter": {**adapter_changes, "sha256": digest(saved_adapter_path)},
            "head": {**head_changes, "sha256": digest(args.checkpoint / "head.pt")},
            "changed_adapter_tensor_count": adapter_changes["changed_tensor_count"],
            "changed_head_tensor_count": head_changes["changed_tensor_count"],
            "dtypes": {"backbone": dtypes["frozen_parameter_dtype_counts"],
                       "lora": dtypes["lora_trainable_parameter_dtype_counts"], "head": head_dtypes},
            "training_metrics": metrics,
            "training_timing": timing,
            "reload_peak_allocated_bytes": allocated_bytes("cuda"),
            "reload_peak_reserved_bytes": torch.cuda.max_memory_reserved("cuda"),
            "tokenizer_load_seconds": tokenizer_load_seconds,
            "model_construction_seconds": model_load_seconds,
            "warm_start_seconds": warm_start_seconds,
            "forward_seconds": reload_forward_seconds,
            "elapsed_seconds": round(time.monotonic() - started, 6),
        })
        set_stage(args.stage_file, "reload_complete", cuda_active=True, status="passed")
        observer.stop()
        result["cuda_observation"] = json.loads(args.observation.read_text(encoding="utf-8"))
        atomic_json(args.output, result)
        return 0
    except BaseException as error:
        result.update({"error_type": type(error).__name__, "error": str(error),
                       "elapsed_seconds": round(time.monotonic() - started, 6)})
        try:
            if "torch" in globals():
                from kev.device import allocated_bytes
                if torch.cuda.is_initialized():
                    observer.capture(torch, allocated_bytes)
        except BaseException as capture_error:
            result["capture_error"] = f"{type(capture_error).__name__}: {capture_error}"
        try:
            observer.stop()
        except BaseException as stop_error:
            result["observation_error"] = f"{type(stop_error).__name__}: {stop_error}"
        try:
            set_stage(args.stage_file, "reload_failed", cuda_active="torch" in globals() and torch.cuda.is_initialized(),
                      error=result["error"])
        except BaseException:
            pass
        try:
            if args.observation.is_file():
                result["cuda_observation"] = json.loads(args.observation.read_text(encoding="utf-8"))
        except BaseException:
            pass
        atomic_json(args.output, result)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
