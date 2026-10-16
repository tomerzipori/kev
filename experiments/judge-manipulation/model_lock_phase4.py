"""Resolve and verify the released Kev-4B parent and Qwen3.5-4B base artifacts."""
import json
import os
from collections import Counter
from pathlib import Path

HF_CACHE = Path(os.environ.get("JUDGE_MANIPULATION_PHASE4_HF_HOME", str(Path.home() / ".cache/huggingface-phase4"))).expanduser().resolve()
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE / "hub")
HF_CACHE.mkdir(parents=True, exist_ok=True)

from huggingface_hub import HfApi, hf_hub_download
from kev.checkpoint import Checkpoint
from kev.suite import digest


ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "runs/release/kev-1.0.json"
BRANCH = ROOT / "experiments/judge-manipulation/BRANCH-phase-4.json"
OUTPUT = ROOT / "experiments/judge-manipulation/model-lock-phase-4.json"
PARENT_ID = "jaredpalmer/kev-4b"
BASE_ID = "Qwen/Qwen3.5-4B-Base"
TOKENIZER_FILES = {
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json",
    "merges.txt", "added_tokens.json", "chat_template.jinja",
}


def sibling_manifest(info):
    result = {}
    for sibling in info.siblings:
        lfs = getattr(sibling, "lfs", None)
        result[sibling.rfilename] = {
            "size": getattr(sibling, "size", None),
            "blob_id": getattr(sibling, "blob_id", None),
            "lfs_sha256": getattr(lfs, "sha256", None) if lfs else None,
        }
    return result


def download_checked(repo, revision, names, remote_files, expected_hashes=None):
    downloaded = {}
    for name in names:
        if name not in remote_files:
            continue
        path = Path(hf_hub_download(repo_id=repo, filename=name, revision=revision))
        actual = digest(path)
        expected = (expected_hashes or {}).get(name)
        remote_sha = remote_files[name]["lfs_sha256"]
        if expected and actual != expected:
            raise RuntimeError(f"{repo}@{revision}:{name} SHA-256 {actual} != release SHA-256 {expected}")
        if remote_sha and actual != remote_sha:
            raise RuntimeError(f"{repo}@{revision}:{name} SHA-256 {actual} != Hub LFS SHA-256 {remote_sha}")
        downloaded[name] = {
            **remote_files[name],
            "sha256": actual,
            "verified_release_sha256": expected,
            "local_path": str(path),
        }
    return downloaded


def frozen_file_checks():
    branch = json.loads(BRANCH.read_text(encoding="utf-8"))
    checks = {}
    for name, expected in branch["frozen_files"].items():
        path = ROOT / name
        actual = digest(path)
        checks[name] = {"expected_sha256": expected, "actual_sha256": actual, "ok": actual == expected}
        if actual != expected:
            raise RuntimeError(f"frozen file changed: {name} has {actual}, expected {expected}")
    plan = checks.get("MASTER-PLAN.md")
    if not plan or plan["actual_sha256"] != "812f13d43f41e05597b09fd783a7486aa7999f80fd995db94815b826c17892ba":
        raise RuntimeError("MASTER-PLAN.md does not match the registered Phase 4 source-of-truth hash")
    return checks


def main():
    release = json.loads(RELEASE.read_text(encoding="utf-8"))
    released = release["hub"]["repos"][PARENT_ID]
    parent_revision = released["weights_revision"]
    base_revision = released["head"]["base_revision"]
    if released["head"]["base"] != BASE_ID:
        raise RuntimeError(f"Kev-4B release points at unexpected base {released['head']['base']!r}")

    api = HfApi()
    parent_info = api.model_info(PARENT_ID, revision=parent_revision, files_metadata=True)
    base_info = api.model_info(BASE_ID, revision=base_revision, files_metadata=True)
    if parent_info.sha != parent_revision:
        raise RuntimeError(f"Kev-4B resolved to {parent_info.sha}, expected {parent_revision}")
    if base_info.sha != base_revision:
        raise RuntimeError(f"Qwen base resolved to {base_info.sha}, expected {base_revision}")

    parent_files = sibling_manifest(parent_info)
    base_files = sibling_manifest(base_info)
    parent_downloads = download_checked(
        PARENT_ID, parent_revision,
        {"adapter_config.json", "adapter_model.safetensors", "head.pt", *TOKENIZER_FILES},
        parent_files, released["weights_sha256"],
    )
    base_downloads = download_checked(
        BASE_ID, base_revision,
        {"config.json", *TOKENIZER_FILES}, base_files,
    )
    for required in ("adapter_config.json", "adapter_model.safetensors", "head.pt", "tokenizer.json"):
        if required not in parent_downloads:
            raise RuntimeError(f"Kev-4B checkpoint is missing {required}")
    for required in ("config.json", "tokenizer.json"):
        if required not in base_downloads:
            raise RuntimeError(f"Qwen base is missing {required}")

    adapter_config = json.loads(Path(parent_downloads["adapter_config.json"]["local_path"]).read_text(encoding="utf-8"))
    actual_targets = set(adapter_config.get("target_modules", []))
    if not actual_targets:
        raise RuntimeError("Kev-4B adapter_config.json does not declare target_modules")
    if int(adapter_config.get("r", -1)) != 16 or int(adapter_config.get("lora_alpha", -1)) != 32:
        raise RuntimeError("Kev-4B LoRA rank/alpha do not match rank 16, alpha 32")

    parent_meta = Checkpoint(Path(parent_downloads["head.pt"]["local_path"]).parent).meta
    if parent_meta.base != BASE_ID or parent_meta.base_revision != base_revision:
        raise RuntimeError("Kev-4B head.pt base name/revision differs from the release record")
    if parent_meta.lora != 16 or parent_meta.head_dim != 256:
        raise RuntimeError("Kev-4B head.pt architecture differs from the expected rank-16, 256-dim head")
    if parent_meta.weights != "lora" or parent_meta.option_isolation or parent_meta.special_embeddings:
        raise RuntimeError("Kev-4B parent architecture is not the expected standard LoRA checkpoint")

    base_config = json.loads(Path(base_downloads["config.json"]["local_path"]).read_text(encoding="utf-8"))
    text_config = base_config.get("text_config", base_config)
    layer_types = text_config.get("layer_types", [])
    architecture = {
        "model_type": base_config.get("model_type"),
        "architectures": base_config.get("architectures"),
        "text_model_type": text_config.get("model_type"),
        "hidden_size": text_config.get("hidden_size"),
        "num_hidden_layers": text_config.get("num_hidden_layers"),
        "vocab_size": text_config.get("vocab_size"),
        "max_position_embeddings": text_config.get("max_position_embeddings"),
        "layer_type_counts": dict(Counter(layer_types)),
        "hybrid": "linear_attention" in set(layer_types),
    }
    if architecture["hidden_size"] != 2560 or architecture["num_hidden_layers"] != 32 or not architecture["hybrid"]:
        raise RuntimeError(f"Qwen config architecture did not match the released hybrid 4B base: {architecture}")

    model_files = {
        name: metadata for name, metadata in base_files.items()
        if name.endswith(".safetensors") or name.endswith(".safetensors.index.json")
    }
    result = {
        "schema_version": 1,
        "status": "locked",
        "recorded_on": "2026-10-15",
        "selection": {
            "user_confirmation": "The user confirmed Kev-4B as the target model and the released Kev-4B as its baseline, following the current objective.",
            "resolution": "This lock resolves stale Kev-0.8B recipe/baseline text for Phase 4 without editing frozen Phase 0/1 files or MASTER-PLAN.md.",
        },
        "parent": {
            "repo": PARENT_ID,
            "weights_revision": parent_revision,
            "release_card_commit": released["card_commit"],
            "resolved_commit": parent_info.sha,
            "head": {key: getattr(parent_meta, key) for key in ("base", "base_revision", "lora", "head_dim", "option_isolation", "special_embeddings", "weights_dtype", "temperature", "weights")},
            "adapter": {
                "rank": adapter_config["r"],
                "alpha": adapter_config["lora_alpha"],
                "dropout": adapter_config.get("lora_dropout"),
                "target_modules": sorted(actual_targets),
            },
            "files": parent_downloads,
        },
        "base": {
            "repo": BASE_ID,
            "revision": base_revision,
            "resolved_commit": base_info.sha,
            "architecture": architecture,
            "config_and_tokenizer_files": base_downloads,
            "weight_file_metadata": model_files,
        },
        "recipe": {
            "base_weights_dtype": "bf16",
            "autocast_dtype": "bf16",
            "lora_rank": 16,
            "lora_targets": "all",
            "adapter_and_head_dtype": "fp32, verified at runtime",
            "batch": 1,
            "accumulation": 16,
            "gradient_checkpointing": True,
            "max_state_tokens_including_state_delimiter": 2048,
            "loss": "ordinary cross-entropy",
            "augmentations": {"p_none": 0, "p_none_distract": 0, "p_distract": 0, "p_none_pair": 0},
            "full_finetuning": False,
            "quantization": None,
            "attention": "PyTorch SDPA",
            "serving_fused_kernels": False,
            "cuda_graphs": False,
        },
        "frozen_file_checks": frozen_file_checks(),
    }
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if OUTPUT.exists():
        existing = json.loads(OUTPUT.read_text(encoding="utf-8"))
        def stable(value):
            if isinstance(value, dict):
                return {key: stable(item) for key, item in value.items() if key not in {"local_path", "downloaded_sha256", "downloaded_sha256_matches_hub_lfs"}}
            if isinstance(value, list):
                return [stable(item) for item in value]
            return value
        if stable(existing) != stable(result):
            raise RuntimeError("model-lock-phase-4.json already exists and differs; refusing to overwrite it")
        print(f"Model lock already matches verified artifacts; left unchanged: {OUTPUT}")
        return
    OUTPUT.write_text(serialized, encoding="utf-8")
    print(json.dumps({"model_lock": str(OUTPUT), "parent_revision": parent_revision, "base_revision": base_revision,
                      "adapter_targets": sorted(actual_targets), "base_architecture": architecture}, indent=2))


if __name__ == "__main__":
    main()
