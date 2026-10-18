"""Check the two optional Qwen3.5 CUDA kernels against Transformers 5.17 references."""

import argparse
import importlib.metadata
import inspect
import json
import math
import os
import platform
import sys
import sysconfig
import time
from datetime import datetime, timezone
from pathlib import Path

from phase4_shared_memory_support import CudaObservation, atomic_json, set_stage


def _error(reference, optimized):
    delta = reference.detach().float() - optimized.detach().float()
    return {"max_abs": float(delta.abs().max()), "rms": float(delta.square().mean().sqrt())}


def _compare_gradients(reference, optimized, names):
    results = {}
    for name, left, right in zip(names, reference, optimized, strict=True):
        if left.grad is None or right.grad is None:
            raise RuntimeError(f"{name} did not receive a backward gradient")
        for label, gradient in (("reference", left.grad), ("optimized", right.grad)):
            if not torch.isfinite(gradient).all():
                raise RuntimeError(f"{name} has a non-finite {label} gradient")
            norm = float(gradient.float().norm())
            if not math.isfinite(norm) or norm <= 0:
                raise RuntimeError(f"{name} has a zero or non-finite {label} gradient norm")
        torch.testing.assert_close(left.grad, right.grad, atol=0.02, rtol=0.05, msg=f"{name} gradient")
        results[name] = {**_error(left.grad, right.grad), "gradient_norm": float(right.grad.float().norm())}
    return results


def _clone_leaves(values):
    return [value.detach().clone().requires_grad_(True) for value in values]


def _delta_check(reference_fn, optimized_fn, length):
    generator = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn((1, length, 16, 128), generator=generator, device="cuda", dtype=torch.bfloat16) * 0.1
    k = torch.randn((1, length, 16, 128), generator=generator, device="cuda", dtype=torch.bfloat16) * 0.1
    v = torch.randn((1, length, 32, 128), generator=generator, device="cuda", dtype=torch.bfloat16) * 0.1
    g = -torch.rand((1, length, 32), generator=generator, device="cuda", dtype=torch.float32) * 0.1
    beta_logits = torch.randn((1, length, 32), generator=generator, device="cuda", dtype=torch.bfloat16) * 0.1
    upstream = torch.randn((1, length, 32, 128), generator=generator, device="cuda", dtype=torch.bfloat16)
    ref_q, ref_k, ref_v, ref_g, ref_b = _clone_leaves((q, k, v, g, beta_logits))
    opt_q, opt_k, opt_v, opt_g, opt_b = _clone_leaves((q, k, v, g, beta_logits))

    def run(fn, inputs):
        q0, k0, v0, g0, b0 = inputs
        q0 = q0.repeat_interleave(2, dim=2)
        k0 = k0.repeat_interleave(2, dim=2)
        out, final_state = fn(q0, k0, v0, g=g0, beta=b0.sigmoid(), output_final_state=False,
                              use_qk_l2norm_in_kernel=True)
        if final_state is not None:
            raise RuntimeError("kernel returned a recurrent state although output_final_state=False")
        return out

    ref_out = run(reference_fn, (ref_q, ref_k, ref_v, ref_g, ref_b))
    opt_out = run(optimized_fn, (opt_q, opt_k, opt_v, opt_g, opt_b))
    if not torch.isfinite(ref_out).all() or not torch.isfinite(opt_out).all():
        raise RuntimeError("gated delta rule produced non-finite output")
    torch.testing.assert_close(ref_out, opt_out, atol=0.01, rtol=0.02, msg="gated delta rule output")
    (ref_out.float() * upstream.float()).sum().backward()
    (opt_out.float() * upstream.float()).sum().backward()
    gradients = _compare_gradients(
        (ref_q, ref_k, ref_v, ref_g, ref_b), (opt_q, opt_k, opt_v, opt_g, opt_b),
        ("query", "key", "value", "decay", "beta_logits"),
    )
    return {"length": length, "output": _error(ref_out, opt_out), "gradients": gradients}


def _convolution_check(reference_fn, optimized_fn, length):
    generator = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn((1, 8192, length), generator=generator, device="cuda", dtype=torch.bfloat16) * 0.1
    weight = torch.randn((8192, 4), generator=generator, device="cuda", dtype=torch.bfloat16) * 0.1
    ref_x, ref_weight = _clone_leaves((x, weight))
    opt_x, opt_weight = _clone_leaves((x, weight))
    ref_out = reference_fn(ref_x, ref_weight, bias=None, activation="silu")
    opt_out = optimized_fn(opt_x, opt_weight, bias=None, activation="silu")
    if not torch.isfinite(ref_out).all() or not torch.isfinite(opt_out).all():
        raise RuntimeError("causal convolution produced non-finite output")
    torch.testing.assert_close(ref_out, opt_out, atol=0.01, rtol=0.02, msg="causal convolution output")
    upstream = torch.randn(ref_out.shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    (ref_out.float() * upstream.float()).sum().backward()
    (opt_out.float() * upstream.float()).sum().backward()
    gradients = _compare_gradients((ref_x, ref_weight), (opt_x, opt_weight), ("input", "weight"))
    return {"length": length, "output": _error(ref_out, opt_out), "gradients": gradients, "bias": "not present in Qwen3.5 GatedDeltaNet.conv1d"}


def _reference_identity(fn, module):
    source = Path(inspect.getsourcefile(fn)).resolve()
    module_path = Path(module.__file__).resolve()
    if fn.__module__ != module.__name__ or source != module_path:
        raise RuntimeError(f"unwrapped reference does not come from installed Transformers source: {fn.__module__} {source}")
    return {"module": fn.__module__, "qualname": fn.__qualname__, "source": str(source)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--overlay", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage-file", required=True, type=Path)
    parser.add_argument("--observation", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    observer = None
    result = {"status": "failed", "started_utc": datetime.now(timezone.utc).isoformat()}
    try:
        set_stage(args.stage_file, "kernel_correctness_pre_cuda_import", cuda_active=False)
        observer = CudaObservation(args.observation, "kernel_correctness")
        observer.start()
        import torch
        globals()["torch"] = torch
        from kev.device import allocated_bytes
        import transformers
        from transformers import AutoConfig
        import transformers.models.qwen3_5.modeling_qwen3_5 as qwen
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        from causal_conv1d import causal_conv1d_fn

        if transformers.__version__ != "5.17.0":
            raise RuntimeError(f"expected Transformers 5.17.0, found {transformers.__version__}")
        overlay = args.overlay.resolve()
        imported_paths = {
            "fla": str(Path(importlib.import_module("fla").__file__).resolve()),
            "causal_conv1d": str(Path(importlib.import_module("causal_conv1d").__file__).resolve()),
        }
        for name, path in imported_paths.items():
            if not Path(path).is_relative_to(overlay):
                raise RuntimeError(f"{name} imported outside the registered overlay: {path}")

        config = AutoConfig.from_pretrained(args.base, revision=args.base_revision, local_files_only=True)
        text_config = getattr(config, "text_config", config)
        actual = {
            "key_heads": text_config.linear_num_key_heads,
            "value_heads": text_config.linear_num_value_heads,
            "key_dim": text_config.linear_key_head_dim,
            "value_dim": text_config.linear_value_head_dim,
            "convolution_width": text_config.linear_conv_kernel_dim,
        }
        actual["convolution_channels"] = actual["key_heads"] * actual["key_dim"] * 2 + actual["value_heads"] * actual["value_dim"]
        expected = {"key_heads": 16, "value_heads": 32, "key_dim": 128, "value_dim": 128,
                    "convolution_width": 4, "convolution_channels": 8192}
        if actual != expected:
            raise RuntimeError(f"pinned Qwen3.5 config differs from the CPU-reviewed shapes: {actual}")

        ref_delta = inspect.unwrap(qwen.torch_chunk_gated_delta_rule)
        ref_conv = inspect.unwrap(qwen.causal_conv1d_fn)
        references = {
            "delta_rule": _reference_identity(ref_delta, qwen),
            "causal_conv1d": _reference_identity(ref_conv, qwen),
        }
        optimized = {
            "delta_rule": {"module": chunk_gated_delta_rule.__module__, "qualname": chunk_gated_delta_rule.__qualname__,
                           "source": str(Path(inspect.getsourcefile(chunk_gated_delta_rule)).resolve())},
            "causal_conv1d": {"module": causal_conv1d_fn.__module__, "qualname": causal_conv1d_fn.__qualname__,
                              "source": str(Path(inspect.getsourcefile(causal_conv1d_fn)).resolve())},
        }
        if not optimized["delta_rule"]["module"].startswith("fla.") or not optimized["causal_conv1d"]["module"].startswith("causal_conv1d."):
            raise RuntimeError(f"optimized functions are not the optional package implementations: {optimized}")

        set_stage(args.stage_file, "kernel_correctness_device_inventory", cuda_active=False)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("the registered single CUDA device is unavailable")
        torch.cuda.set_device(0)
        properties = torch.cuda.get_device_properties(0)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA device does not support BF16")
        observer.capture(torch, allocated_bytes)
        device = {"name": properties.name, "total_memory_bytes": properties.total_memory,
                  "compute_capability": [properties.major, properties.minor], "bf16_supported": True}

        checks = []
        for length in (128, 2099):
            set_stage(args.stage_file, f"kernel_correctness_delta_{length}", cuda_active=True)
            t0 = time.monotonic()
            delta = _delta_check(ref_delta, chunk_gated_delta_rule, length)
            torch.cuda.synchronize()
            delta["elapsed_seconds"] = round(time.monotonic() - t0, 6)
            set_stage(args.stage_file, f"kernel_correctness_conv_{length}", cuda_active=True)
            t0 = time.monotonic()
            conv = _convolution_check(ref_conv, causal_conv1d_fn, length)
            torch.cuda.synchronize()
            conv["elapsed_seconds"] = round(time.monotonic() - t0, 6)
            checks.append({"delta_rule": delta, "convolution": conv})
        observer.capture(torch, allocated_bytes)
        observer.stop()
        observation = json.loads(args.observation.read_text(encoding="utf-8"))
        if observation.get("cuda_active") is not True:
            raise RuntimeError("the in-process allocator observation never recorded an initialized CUDA context")
        result.update({
            "status": "passed",
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
            "python": platform.python_version(),
            "python_abi": sysconfig.get_config_var("SOABI"),
            "cuda_device": device,
            "config_shapes": actual,
            "overlay_path": str(overlay),
            "reference_functions": references,
            "optimized_functions": optimized,
            "package_versions": {name: importlib.metadata.version(name) for name in
                                 ("flash-linear-attention", "fla-core", "causal-conv1d", "triton")},
            "imported_paths": imported_paths,
            "checks": checks,
            "cuda_observation": observation,
            "elapsed_seconds": round(time.monotonic() - started, 6),
        })
        set_stage(args.stage_file, "kernel_correctness_complete", cuda_active=False, status="passed")
        atomic_json(args.output, result)
        return 0
    except BaseException as error:
        result.update({"error_type": type(error).__name__, "error": str(error),
                       "elapsed_seconds": round(time.monotonic() - started, 6)})
        if observer:
            try:
                if "torch" in globals() and "allocated_bytes" in locals():
                    observer.capture(torch, allocated_bytes)
                observer.stop()
            except BaseException as stop_error:
                result["observation_error"] = f"{type(stop_error).__name__}: {stop_error}"
        try:
            if args.observation.is_file():
                result["cuda_observation"] = json.loads(args.observation.read_text(encoding="utf-8"))
        except BaseException as read_error:
            result["observation_read_error"] = f"{type(read_error).__name__}: {read_error}"
        atomic_json(args.output, result)
        set_stage(args.stage_file, "kernel_correctness_failed", cuda_active=False, error=result["error"])
        raise


if __name__ == "__main__":
    raise SystemExit(main())
