"""Run kev.train unchanged while saving the final in-memory output for reload parity."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from kev import train
from kev.checkpoint import Checkpoint
from kev.data import load_records, materialize
from kev.device import allocated_bytes
from kev.model import DecisionModel as OriginalDecisionModel, training_context
from kev.predictors import kernel_environment
from phase4_model_evidence import parameter_dtype_evidence


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--phase4-fixture", required=True)
parser.add_argument("--phase4-proof-dir", required=True)
phase4, train_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *train_args]
fixture_path = Path(phase4.phase4_fixture).resolve()
proof_dir = Path(phase4.phase4_proof_dir).resolve()
proof_dir.mkdir(parents=True, exist_ok=True)
instances = []
stage_path = proof_dir / "stage.json"


def set_stage(stage, **details):
    temporary = stage_path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"stage": stage, "updated_utc": datetime.now(timezone.utc).isoformat(), **details}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(stage_path)


set_stage("model_load")


class CapturedDecisionModel(OriginalDecisionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        instances.append(self)


def logits_as_lists(result):
    return [[tensor.detach().float().cpu().tolist() for tensor in row] for row in result]


def instrumented_finish(out, meta, tok):
    model = instances[-1]
    request = load_records(fixture_path)[0]
    rec = materialize(request)
    context = training_context(2048)
    enc = model.encode(tok, rec, strict=True, max_state=context["max_state"], max_branch=context["max_branch"])
    model.eval()
    set_stage("final_proof_forward")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        before = model.forward_batch([enc], shared_prefix=False)
    set_stage("final_proof_forward_complete")
    pre_path = proof_dir / "pre-save-output.pt"
    torch.save({"logits": logits_as_lists(before), "state_tokens": enc["state_tokens"],
                "encoded_tokens": len(enc["ids"]), "branch_tokens": len(enc["ids"]) - enc["state_tokens"]}, pre_path)

    dtype_evidence = parameter_dtype_evidence(model)
    lora_dtypes = sorted(dtype_evidence["lora_trainable_parameter_dtype_counts"])
    backbone_dtypes = sorted(dtype_evidence["frozen_parameter_dtype_counts"])
    head_dtypes = sorted({str(param.dtype).removeprefix("torch.") for param in model.head.parameters()})
    if head_dtypes != ["float32"]:
        raise RuntimeError(f"pointer head parameters must remain FP32, found {head_dtypes}")
    previous = json.loads((proof_dir / "in_training_model.json").read_text(encoding="utf-8"))
    architecture = {
        **previous,
        "attention_implementation": getattr(model.lm.config, "_attn_implementation", None),
        "hybrid": model.hybrid,
        "lora_trainable_parameter_count": sum(param.numel() for name, param in model.lm.named_parameters() if "lora_" in name.lower() and param.requires_grad),
        "lora_trainable_dtypes": lora_dtypes,
        "parameter_dtype_evidence": dtype_evidence,
        "backbone_dtypes": backbone_dtypes,
        "head_dtypes": head_dtypes,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "peak_allocated_bytes": allocated_bytes("cuda"),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved("cuda"),
    }
    set_stage("checkpoint_save")
    train.finish_checkpoint_original(out, meta, tok)
    (proof_dir / "in_training_model.json").write_text(json.dumps(architecture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    set_stage("checkpoint_saved", checkpoint=str(out))
    return None


train.DecisionModel = CapturedDecisionModel
train.finish_checkpoint_original = train.finish_checkpoint
train.finish_checkpoint = instrumented_finish

original_warm_start = Checkpoint.warm_start


def instrumented_warm_start(checkpoint, model, ours):
    set_stage("warm_start")
    result = original_warm_start(checkpoint, model, ours)
    dtype_evidence = parameter_dtype_evidence(model)
    lora_dtypes = sorted(dtype_evidence["lora_trainable_parameter_dtype_counts"])
    backbone_dtypes = sorted(dtype_evidence["frozen_parameter_dtype_counts"])
    head_dtypes = sorted({str(param.dtype).removeprefix("torch.") for param in model.head.parameters()})
    if head_dtypes != ["float32"]:
        raise RuntimeError(f"unexpected training dtypes after warm start: backbone={backbone_dtypes}, LoRA={lora_dtypes}, head={head_dtypes}")
    environment = kernel_environment(model, "cuda")
    evidence = {
        "status": "model_loaded_and_warm_started",
        "base": ours.base,
        "base_revision": ours.base_revision,
        "lora_rank": ours.lora,
        "head_dim": ours.head_dim,
        "lora_targets_argument": "all",
        "warm_start_coverage": result,
        "attention_implementation": getattr(model.lm.config, "_attn_implementation", None),
        "kernel_environment": environment,
        "hybrid": model.hybrid,
        "lora_trainable_parameter_count": sum(param.numel() for name, param in model.lm.named_parameters() if "lora_" in name.lower() and param.requires_grad),
        "lora_trainable_dtypes": lora_dtypes,
        "parameter_dtype_evidence": dtype_evidence,
        "backbone_dtypes": backbone_dtypes,
        "head_dtypes": head_dtypes,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "peak_allocated_bytes_so_far": allocated_bytes("cuda"),
        "peak_reserved_bytes_so_far": torch.cuda.max_memory_reserved("cuda"),
    }
    (proof_dir / "in_training_model.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    set_stage("warm_start_complete", warm_start_coverage=result)
    return result


Checkpoint.warm_start = instrumented_warm_start

original_batch_loss = train.batch_loss


def instrumented_batch_loss(*args, **kwargs):
    set_stage("forward")
    try:
        result = original_batch_loss(*args, **kwargs)
    except Exception:
        set_stage("forward_failed")
        raise
    set_stage("forward_complete")
    return result


train.batch_loss = instrumented_batch_loss
original_tensor_backward = torch.Tensor.backward


def instrumented_backward(self, *args, **kwargs):
    set_stage("backward")
    try:
        result = original_tensor_backward(self, *args, **kwargs)
    except Exception:
        set_stage("backward_failed")
        raise
    set_stage("backward_complete")
    return result


torch.Tensor.backward = instrumented_backward
original_clip_grad = torch.nn.utils.clip_grad_norm_


def instrumented_clip_grad(*args, **kwargs):
    set_stage("gradient_clip")
    try:
        result = original_clip_grad(*args, **kwargs)
    except Exception:
        set_stage("gradient_clip_failed")
        raise
    set_stage("gradient_clip_complete")
    return result


torch.nn.utils.clip_grad_norm_ = instrumented_clip_grad
original_optimizer_step = torch.optim.AdamW.step


def instrumented_optimizer_step(self, *args, **kwargs):
    set_stage("optimizer_step")
    try:
        result = original_optimizer_step(self, *args, **kwargs)
    except Exception:
        set_stage("optimizer_step_failed")
        raise
    set_stage("optimizer_step_complete")
    return result


torch.optim.AdamW.step = instrumented_optimizer_step

try:
    train.main()
except BaseException as error:
    # Do not initialize CUDA to collect evidence for an import/argument failure.
    # If the measured model process already initialized it, preserve allocator
    # peaks and the driver's free/total reading at the actual failure stage.
    failure = {"error_type": type(error).__name__, "error": str(error), "cuda_initialized": False}
    try:
        if stage_path.is_file():
            failure["stage"] = json.loads(stage_path.read_text(encoding="utf-8"))
    except BaseException as evidence_error:
        failure["stage_evidence_error"] = f"{type(evidence_error).__name__}: {evidence_error}"
    try:
        failure["cuda_initialized"] = torch.cuda.is_initialized()
    except BaseException as evidence_error:
        failure["cuda_initialized_error"] = f"{type(evidence_error).__name__}: {evidence_error}"
    if failure["cuda_initialized"]:
        for name, measure in (
            ("cuda_free_bytes_at_failure", lambda: torch.cuda.mem_get_info("cuda")[0]),
            ("cuda_total_bytes_at_failure", lambda: torch.cuda.mem_get_info("cuda")[1]),
            ("peak_allocated_bytes", lambda: allocated_bytes("cuda")),
            ("peak_reserved_bytes", lambda: torch.cuda.max_memory_reserved("cuda")),
        ):
            try:
                failure[name] = measure()
            except BaseException as evidence_error:
                failure[f"{name}_error"] = f"{type(evidence_error).__name__}: {evidence_error}"
    try:
        (proof_dir / "training-failure-memory.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except BaseException:
        pass
    raise
