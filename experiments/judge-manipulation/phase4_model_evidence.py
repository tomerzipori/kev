"""Small, read-only evidence helpers for the Phase 4 local feasibility probe."""
from collections import Counter


def parameter_dtype_evidence(model):
    """Report backbone precision without recasting upstream Qwen stability parameters."""
    modules = dict(model.lm.named_modules())
    lora = []
    frozen = []
    for name, parameter in model.lm.named_parameters():
        dtype = str(parameter.dtype).removeprefix("torch.")
        if "lora_" in name.lower():
            if parameter.requires_grad:
                lora.append((name, dtype))
            continue
        frozen.append((name, dtype))

    fp32_stability = []
    invalid = []
    for name, dtype in frozen:
        if dtype == "bfloat16":
            continue
        owner_name, _, parameter_name = name.rpartition(".")
        owner = modules.get(owner_name)
        owner_type = type(owner).__name__ if owner is not None else None
        # Transformers 5.17.0 Qwen3.5 computes A_log/dt_bias gate math in FP32 and
        # promotes gated RMSNorm activations to FP32. Preserve these upstream
        # parameters as loaded; never cast them in this probe.
        stability_parameter = (
            owner_type == "Qwen3_5GatedDeltaNet" and parameter_name in {"A_log", "dt_bias"}
        ) or (
            owner_type == "Qwen3_5RMSNormGated" and parameter_name == "weight"
        )
        if dtype == "float32" and stability_parameter:
            fp32_stability.append(name)
        else:
            invalid.append({"name": name, "dtype": dtype, "owner_type": owner_type})

    if invalid:
        raise RuntimeError(f"unexpected frozen backbone parameter dtype(s): {invalid[:12]}")
    lora_dtypes = sorted({dtype for _, dtype in lora})
    if lora_dtypes != ["float32"]:
        raise RuntimeError(f"LoRA parameters must remain FP32, found {lora_dtypes}")
    return {
        "frozen_parameter_dtype_counts": dict(sorted(Counter(dtype for _, dtype in frozen).items())),
        "frozen_parameter_count": len(frozen),
        "frozen_fp32_parameter_names": sorted(name for name, dtype in frozen if dtype == "float32"),
        "upstream_stability_fp32_parameter_names": sorted(fp32_stability),
        "lora_trainable_parameter_dtype_counts": dict(sorted(Counter(dtype for _, dtype in lora).items())),
        "lora_trainable_parameter_count": len(lora),
    }
