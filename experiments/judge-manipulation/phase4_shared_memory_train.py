"""Run the registered two-update Kev trainer and record reload-parity inputs."""

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

from phase4_shared_memory_support import CudaObservation, atomic_json, set_stage


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--phase4-fixture", required=True)
parser.add_argument("--phase4-proof-dir", required=True)
parser.add_argument("--phase4-observation", required=True)
parser.add_argument("--phase4-overlay", required=True)
phase4, train_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *train_args]
training_args = {key: train_args[train_args.index(key) + 1] for key in
                 ("--base", "--base_revision", "--lora", "--lora_targets")}
accumulation = int(train_args[train_args.index("--accum") + 1])
if accumulation != 16:
    raise RuntimeError(f"registered accumulation is 16, got {accumulation}")
fixture_path = Path(phase4.phase4_fixture).resolve()
proof_dir = Path(phase4.phase4_proof_dir).resolve()
proof_dir.mkdir(parents=True, exist_ok=True)
stage_path = proof_dir / "training-stage.json"
observation_path = Path(phase4.phase4_observation)
overlay_path = str(Path(phase4.phase4_overlay).resolve())
instances = []
microbatches = 0
optimizer_update_seconds = []
optimizer_window_seconds = []
update_gradient_norms = []
optimizer_updates = 0
optimizer_window_started = None
tokenizer_load_seconds = None
model_construction_seconds = None
warm_start_seconds = []
checkpoint_save_seconds = None
started = time.monotonic()

set_stage(stage_path, "training_pre_cuda_import", cuda_active=False)
observer = CudaObservation(observation_path, "training")
observer.start()
import torch
globals()["torch"] = torch

set_stage(stage_path, "training_imports", cuda_active=False)
import kev.train as train
from kev.checkpoint import Checkpoint
from kev.data import load_records, materialize
from kev.device import allocated_bytes
from kev.model import DecisionModel as OriginalDecisionModel, training_context
from kev.predictors import kernel_environment
from phase4_model_evidence import parameter_dtype_evidence


class CapturedDecisionModel(OriginalDecisionModel):
    def __init__(self, *args, **kwargs):
        global model_construction_seconds
        started_model = time.monotonic()
        super().__init__(*args, **kwargs)
        model_construction_seconds = round(time.monotonic() - started_model, 6)
        instances.append(self)


def logits_as_lists(result):
    return [[tensor.detach().float().cpu().tolist() for tensor in row] for row in result]


def _eval_state(model):
    return {"model_training": model.training, "backbone_training": model.lm.training,
            "head_training": model.head.training}


def _lora_ranks(model):
    configs = getattr(model.lm, "peft_config", {})
    return {name: int(config.r) for name, config in configs.items()}


def _arg(name):
    index = train_args.index(name)
    return train_args[index + 1]


def _kernel_binding(model, name):
    layer = next((item for item in model.lm.modules() if type(item).__name__.endswith("GatedDeltaNet")), None)
    if layer is None:
        raise RuntimeError("loaded base has no GatedDeltaNet layer to prove kernel bindings")
    module = sys.modules[type(layer).__module__]
    bound = getattr(module, name)
    implementation = inspect.getclosurevars(bound).nonlocals.get("implementation", bound) if inspect.isfunction(bound) else bound
    source = inspect.getsourcefile(implementation)
    if source is None:
        raise RuntimeError(f"cannot resolve the installed source for {name}")
    return {"qualified_name": f"{implementation.__module__}.{implementation.__qualname__}",
            "source": str(Path(source).resolve())}


def instrumented_finish(out, meta, tok):
    global checkpoint_save_seconds
    model = instances[-1]
    record = materialize(load_records(fixture_path)[0])
    context = training_context(2048)
    enc = model.encode(tok, record, strict=True, max_state=context["max_state"], max_branch=context["max_branch"])
    if (enc["state_tokens"], len(enc["ids"]) - enc["state_tokens"], len(enc["ids"])) != (2048, 51, 2099):
        raise RuntimeError("final training read does not match the registered token shape")
    model.eval()
    set_stage(stage_path, "training_pre_save_forward", cuda_active=torch.cuda.is_initialized())
    torch.cuda.synchronize()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model.forward_batch([enc], shared_prefix=False)
    torch.cuda.synchronize()
    head_temperature = float(model.head.temperature)
    meta_temperature = float(meta.temperature)
    if head_temperature != meta_temperature:
        raise RuntimeError(f"model head temperature {head_temperature} differs from saved metadata temperature {meta_temperature}")
    pre_save = {
        "logits": logits_as_lists(logits),
        "state_tokens": enc["state_tokens"],
        "encoded_tokens": len(enc["ids"]),
        "branch_tokens": len(enc["ids"]) - enc["state_tokens"],
        "temperature": head_temperature,
        "meta_temperature": meta_temperature,
        "eval_state": _eval_state(model),
        "tf32": {"matmul": torch.backends.cuda.matmul.allow_tf32,
                 "cudnn": torch.backends.cudnn.allow_tf32},
        "kernel_environment": kernel_environment(model, "cuda"),
        "kernel_bindings": {
            "causal_conv1d_fn": _kernel_binding(model, "causal_conv1d_fn"),
            "torch_chunk_gated_delta_rule": _kernel_binding(model, "torch_chunk_gated_delta_rule"),
        },
        "overlay_path": overlay_path,
        "unmerged_lora": _lora_ranks(model) == {"default": 16},
    }
    torch.save(pre_save, proof_dir / "pre-save-output.pt")
    evidence = parameter_dtype_evidence(model)
    head_dtypes = sorted({str(param.dtype).removeprefix("torch.") for param in model.head.parameters()})
    if head_dtypes != ["float32"]:
        raise RuntimeError(f"pointer head must remain FP32, found {head_dtypes}")
    lora_dtypes = sorted(evidence["lora_trainable_parameter_dtype_counts"])
    if lora_dtypes != ["float32"]:
        raise RuntimeError(f"LoRA parameters must remain FP32, found {lora_dtypes}")
    model_proof = {
        "status": "loaded_and_warm_started",
        "base": training_args["--base"],
        "base_revision": training_args["--base_revision"],
        "init_from": _arg("--init_from"),
        "lora_ranks": _lora_ranks(model),
        "lora_targets_argument": training_args["--lora_targets"],
        "option_isolation": model.option_isolation,
        "special_embeddings": bool(model.special_embeddings),
        "full_weight_training": int(_arg("--full_ft")) if "--full_ft" in train_args else False,
        "recipe_arguments": {name: _arg(name) for name in (
            "--head_dim", "--batch", "--accum", "--checkpointing", "--lr", "--lora", "--epochs",
            "--lora_targets", "--dtype", "--weights_dtype", "--max_state", "--seed",
            "--p_none", "--p_none_distract", "--p_distract", "--p_none_pair", "--synthetic_repeat",
            "--public_frac", "--max_steps",
        )},
        "warm_start_coverage": model._phase4_warm_start_coverage,
        "attention_implementation": getattr(model.lm.config, "_attn_implementation", None),
        "hybrid": model.hybrid,
        "kernel_environment": kernel_environment(model, "cuda"),
        "kernel_bindings": pre_save["kernel_bindings"],
        "overlay_path": overlay_path,
        "parameter_dtype_evidence": evidence,
        "head_dtypes": head_dtypes,
        "tf32": pre_save["tf32"],
        "eval_state": pre_save["eval_state"],
        "pre_save_temperature": pre_save["temperature"],
        "pre_save_meta_temperature": pre_save["meta_temperature"],
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "peak_allocated_bytes": allocated_bytes("cuda"),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved("cuda"),
    }
    atomic_json(proof_dir / "training-model.json", model_proof)
    set_stage(stage_path, "checkpoint_save", cuda_active=True)
    save_started = time.monotonic()
    train.finish_checkpoint_original(out, meta, tok)
    checkpoint_save_seconds = round(time.monotonic() - save_started, 6)
    set_stage(stage_path, "checkpoint_saved", cuda_active=True, checkpoint=str(out))
    return None


train.DecisionModel = CapturedDecisionModel
train.finish_checkpoint_original = train.finish_checkpoint
train.finish_checkpoint = instrumented_finish

original_warm_start = Checkpoint.warm_start


def instrumented_warm_start(checkpoint, model, ours):
    set_stage(stage_path, "warm_start", cuda_active=torch.cuda.is_initialized())
    started_warm = time.monotonic()
    result = original_warm_start(checkpoint, model, ours)
    warm_start_seconds.append(round(time.monotonic() - started_warm, 6))
    model._phase4_warm_start_coverage = result
    return result


Checkpoint.warm_start = instrumented_warm_start
original_batch_loss = train.batch_loss


def instrumented_batch_loss(*args, **kwargs):
    global microbatches, optimizer_window_started
    if microbatches % accumulation == 0:
        optimizer_window_started = time.monotonic()
    microbatches += 1
    set_stage(stage_path, "training_forward", cuda_active=torch.cuda.is_initialized(), microbatch=microbatches)
    result = original_batch_loss(*args, **kwargs)
    set_stage(stage_path, "training_forward_complete", cuda_active=True, microbatch=microbatches)
    return result


train.batch_loss = instrumented_batch_loss
original_clip_grad = torch.nn.utils.clip_grad_norm_


def instrumented_clip_grad(*args, **kwargs):
    result = original_clip_grad(*args, **kwargs)
    value = float(result.detach().float().cpu()) if isinstance(result, torch.Tensor) else float(result)
    update_gradient_norms.append(value)
    return result


torch.nn.utils.clip_grad_norm_ = instrumented_clip_grad
original_backward = torch.Tensor.backward


def instrumented_backward(self, *args, **kwargs):
    set_stage(stage_path, "training_backward", cuda_active=True, microbatch=microbatches)
    result = original_backward(self, *args, **kwargs)
    set_stage(stage_path, "training_backward_complete", cuda_active=True, microbatch=microbatches)
    return result


torch.Tensor.backward = instrumented_backward
original_optimizer_step = torch.optim.AdamW.step


def instrumented_optimizer_step(self, *args, **kwargs):
    global optimizer_updates, optimizer_window_started
    set_stage(stage_path, "optimizer_update", cuda_active=True, update=optimizer_updates + 1)
    t0 = time.monotonic()
    result = original_optimizer_step(self, *args, **kwargs)
    torch.cuda.synchronize()
    finished = time.monotonic()
    optimizer_update_seconds.append(round(finished - t0, 6))
    optimizer_window_seconds.append(round(finished - optimizer_window_started, 6) if optimizer_window_started else None)
    optimizer_window_started = None
    optimizer_updates += 1
    set_stage(stage_path, "optimizer_update_complete", cuda_active=True, update=optimizer_updates,
              elapsed_seconds=optimizer_update_seconds[-1])
    return result


torch.optim.AdamW.step = instrumented_optimizer_step

try:
    set_stage(stage_path, "training_model_load", cuda_active=False)
    original_load_tokenizer = train.load_tokenizer

    def instrumented_load_tokenizer(*args, **kwargs):
        global tokenizer_load_seconds
        started_tokenizer = time.monotonic()
        result = original_load_tokenizer(*args, **kwargs)
        tokenizer_load_seconds = round(time.monotonic() - started_tokenizer, 6)
        return result

    train.load_tokenizer = instrumented_load_tokenizer
    train.main()
    observer.capture(torch, allocated_bytes)
    observer.stop()
    out_arg = Path(sys.argv[sys.argv.index("--out") + 1])
    metrics_path = out_arg / "training_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    atomic_json(proof_dir / "training-timing.json", {
        "microbatches": microbatches,
        "optimizer_updates": optimizer_updates,
        "optimizer_operation_seconds": optimizer_update_seconds,
        "optimizer_window_seconds": optimizer_window_seconds,
        "per_update_gradient_norms": update_gradient_norms,
        "tokenizer_load_seconds": tokenizer_load_seconds,
        "model_construction_seconds": model_construction_seconds,
        "warm_start_seconds": warm_start_seconds,
        "checkpoint_save_seconds": checkpoint_save_seconds,
        "training_elapsed_seconds": round(time.monotonic() - started, 6),
        "metrics_path": str(metrics_path.resolve()),
        "metrics": metrics,
        "cuda_observation": json.loads(observation_path.read_text(encoding="utf-8")),
    })
    checkpoint_files = ("adapter_model.safetensors", "head.pt", "training_metrics.json")
    checkpoint_saved = all((out_arg / name).is_file() for name in checkpoint_files)
    if not checkpoint_saved:
        raise RuntimeError(f"checkpoint save is incomplete: {[name for name in checkpoint_files if not (out_arg / name).is_file()]}")
    atomic_json(proof_dir / "training-result.json", {
        "status": "passed",
        "optimizer_updates": optimizer_updates,
        "microbatches": microbatches,
        "checkpoint_saved": checkpoint_saved,
        "checkpoint_path": str(out_arg.resolve()),
        "checkpoint_files": list(checkpoint_files),
        "metrics_path": str(metrics_path.resolve()),
        "timing_path": str((proof_dir / "training-timing.json").resolve()),
    })
    set_stage(stage_path, "training_complete", cuda_active=False, microbatches=microbatches,
              optimizer_updates=optimizer_updates)
except BaseException as error:
    failure = {"error_type": type(error).__name__, "error": str(error), "microbatches": microbatches,
               "optimizer_updates": optimizer_updates, "stage": None,
               "cuda_initialized": torch.cuda.is_initialized()}
    if stage_path.is_file():
        failure["stage"] = json.loads(stage_path.read_text(encoding="utf-8"))
    if failure["cuda_initialized"]:
        failure["peak_allocated_bytes"] = allocated_bytes("cuda")
        failure["peak_reserved_bytes"] = torch.cuda.max_memory_reserved("cuda")
    try:
        observer.stop()
    except BaseException as stop_error:
        failure["observation_error"] = f"{type(stop_error).__name__}: {stop_error}"
    atomic_json(proof_dir / "training-failure.json", failure)
    set_stage(stage_path, "training_failed", cuda_active=failure["cuda_initialized"], error=failure["error"])
    raise

