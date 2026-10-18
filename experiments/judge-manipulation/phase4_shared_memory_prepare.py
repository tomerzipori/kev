"""Verify the approved CPU setup and recreate the registered Phase 4 fixture."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from email.parser import BytesParser
from pathlib import Path


PLAN_SHA256 = "0329f3bc4fc6e9779caf1b32b393729b9f36fc6d28de85bb17e7c1b3bae4b48a"
MODEL_LOCK_SHA256 = "0bdce0513f5d85c09154f6a8808ad973bc2848d679d1144c7093e64dc6d8fb19"
FIXTURE_SHA256 = "ccc4c494c47e2559334b93842ce30c218da0e0fb6234eaf8d111aa2c5c97be20"
OVERLAY = Path("/home/tomer/.cache/kev-phase4-shared/overlays/fla052-triton340")
WHEELS = Path("/home/tomer/.cache/kev-phase4-shared/wheels")
PRIMARY_PYTHON = Path("/mnt/c/Users/Tomer/Documents/GitHub/kev/.venv/bin/python")
EXPECTED_DISTROS = {"flash-linear-attention", "fla-core", "einops", "causal-conv1d", "ninja"}
HISTORICAL_FILES = (
    "experiments/judge-manipulation/model-lock-phase-4.json",
    "experiments/judge-manipulation/report-phase-4.json",
    "experiments/judge-manipulation/report-phase-4.md",
    "experiments/judge-manipulation/feasibility-gate-phase-4.json",
    "experiments/judge-manipulation/feasibility-gate-phase-4.md",
    "experiments/judge-manipulation/memory-gate-stop-phase-4.json",
    "experiments/judge-manipulation/HANDOFF-phase-4.md",
    "experiments/judge-manipulation/AMENDMENT-shared-memory-phase-4.json",
)
SOURCE_FILES = (
    "experiments/judge-manipulation/phase4_shared_memory_prepare.py",
    "experiments/judge-manipulation/phase4_shared_memory_runner.py",
    "experiments/judge-manipulation/phase4_shared_memory_windows_sample.ps1",
    "experiments/judge-manipulation/phase4_shared_memory_gate.py",
    "experiments/judge-manipulation/phase4_shared_memory_kernels.py",
    "experiments/judge-manipulation/phase4_shared_memory_reload.py",
    "experiments/judge-manipulation/phase4_shared_memory_support.py",
    "experiments/judge-manipulation/phase4_shared_memory_train.py",
    "experiments/judge-manipulation/phase4_model_evidence.py",
    "experiments/judge-manipulation/phase4-shared-memory-plan.json",
    "experiments/judge-manipulation/phase4-shared-memory-overlay-lock.json",
    "experiments/judge-manipulation/phase4-shared-memory-overlay-supplement.json",
    "experiments/judge-manipulation/model-lock-phase-4.json",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_local_model_cache(model_lock):
    base = model_lock["base"]
    config_path = Path(base["config_and_tokenizer_files"]["config.json"]["local_path"])
    cache_home = config_path.parents[4].resolve()
    expected_home = (Path.home() / ".cache/huggingface-phase4").resolve()
    if cache_home != expected_home:
        raise RuntimeError(f"locked model assets are outside the registered Phase 4 cache: {cache_home}")
    hub_cache = cache_home / "hub"
    repo_cache = hub_cache / f"models--{base['repo'].replace('/', '--')}" / "snapshots" / base["revision"]
    hub_root = hub_cache.resolve(strict=True)
    verified = []

    def verify(path, spec, label):
        path = Path(path)
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise RuntimeError(f"locked model asset is not a file: {label} ({path})")
        try:
            resolved.relative_to(hub_root)
        except ValueError as error:
            raise RuntimeError(f"locked model asset resolves outside the Phase 4 Hub cache: {label}") from error
        size = resolved.stat().st_size
        if size != spec["size"]:
            raise RuntimeError(f"locked model asset size mismatch for {label}: {size} != {spec['size']}")
        actual_sha = sha256(resolved)
        expected_sha = spec.get("sha256") or spec.get("lfs_sha256")
        if expected_sha and actual_sha != expected_sha:
            raise RuntimeError(f"locked model asset SHA-256 mismatch for {label}: {actual_sha} != {expected_sha}")
        verified.append({"name": label, "path": str(path), "resolved_path": str(resolved),
                         "size_bytes": size, "sha256": actual_sha, "expected_sha256": expected_sha})
        return path

    for name, spec in base["config_and_tokenizer_files"].items():
        path = Path(spec["local_path"])
        if path != repo_cache / name:
            raise RuntimeError(f"locked base asset path does not match its pinned revision: {name}")
        verify(path, spec, f"base/{name}")

    weight_files = base["weight_file_metadata"]
    for name, spec in weight_files.items():
        verify(repo_cache / name, spec, f"base/{name}")
    index_path = repo_cache / "model.safetensors.index.json"
    index = read_json(index_path)
    expected_shards = set(weight_files) - {"model.safetensors.index.json"}
    actual_shards = set(index.get("weight_map", {}).values())
    if actual_shards != expected_shards:
        raise RuntimeError(f"base weight index shard set differs from the model lock: {sorted(actual_shards)}")

    parent = model_lock["parent"]
    for name, spec in parent["files"].items():
        verify(Path(spec["local_path"]), spec, f"parent/{name}")

    environment = {
        "JUDGE_MANIPULATION_PHASE4_HF_HOME": str(cache_home),
        "HF_HOME": str(cache_home),
        "HF_HUB_CACHE": str(hub_cache),
        "HUGGINGFACE_HUB_CACHE": str(hub_cache),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    return {"home": str(cache_home), "hub_cache": str(hub_cache), "environment": environment,
            "base_repo": base["repo"], "base_revision": base["revision"],
            "parent_repo": parent["repo"], "parent_revision": parent["resolved_commit"],
            "verified_assets": verified}


def verify_windows_sampler_contract(experiment_dir):
    script = experiment_dir / "phase4_shared_memory_windows_sample.ps1"
    source = script.read_text(encoding="utf-8")
    fields = ("status", "observed_utc", "error", "windows_available_physical_bytes",
              "windows_adapter_shared_bytes", "windows_adapter_dedicated_bytes",
              "windows_adapter_counters", "windows_gpu_process_counters")
    missing = [name for name in fields
               if not re.search(rf"(?m)^\s*{re.escape(name)}\s*=", source)]
    if missing or "status = 'sampled'" not in source or "status = 'failed'" not in source:
        raise RuntimeError(f"Windows sampler output contract is incomplete: {missing}")
    return {"status": "static_shape_verified", "path": str(script), "sha256": sha256(script),
            "fields": list(fields), "sample_statuses": ["sampled", "failed"],
            "executed_during_preflight": False}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def primary_environment(repo):
    code = r"""
import hashlib, importlib.metadata, json, pathlib, platform, sys, torch
names = ("torch", "transformers", "peft", "triton", "packaging")
items = {}
for name in names:
    dist = importlib.metadata.distribution(name)
    metadata = pathlib.Path(dist._path) / "METADATA"
    items[name] = {"version": dist.version, "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest()}
result = {
    "python": platform.python_version(), "executable": str(pathlib.Path(sys.executable).resolve()),
    "packages": items, "torch_runtime_version": torch.__version__, "torch_cuda_build": torch.version.cuda,
    "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
    "cuda_initialized": torch.cuda.is_initialized(),
}
print(json.dumps(result, sort_keys=True))
"""
    result = subprocess.run([str(PRIMARY_PYTHON), "-c", code], cwd=repo, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode:
        raise RuntimeError(f"primary Python inventory failed: {result.stderr.strip()}")
    try:
        values = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"primary Python inventory returned invalid JSON: {result.stdout!r}") from error
    if Path(values["executable"]) != PRIMARY_PYTHON.resolve():
        raise RuntimeError(f"unexpected primary Python executable: {values['executable']}")
    if values["cuda_initialized"]:
        raise RuntimeError("CPU preflight unexpectedly initialized a CUDA context")
    if values["python"] != "3.13.16" or values["torch_runtime_version"] != "2.8.0+cu128":
        raise RuntimeError(f"primary Python runtime differs from the pinned environment: {values}")
    expected = {"transformers": "5.17.0", "peft": "0.21.0", "triton": "3.4.0"}
    for name, version in expected.items():
        if values["packages"][name]["version"] != version:
            raise RuntimeError(f"primary {name} must remain {version}: {values['packages'][name]}")
    return values


def overlay_inventory(overlay_lock_path, supplement_path):
    lock = read_json(overlay_lock_path)
    supplement = read_json(supplement_path)
    distributions = lock["overlay_distributions"]
    wheels = []
    for wheel in lock["wheels"]:
        filename = wheel["filename"].lower()
        matches = [name for name in distributions
                   if filename.startswith(re.sub(r"[-_.]+", "_", name).lower() + "-")]
        if len(matches) != 1:
            raise RuntimeError(f"cannot map approved wheel to one distribution: {wheel['filename']}")
        name = matches[0]
        distribution = distributions[name]
        wheels.append({**wheel, "name": name, "version": distribution["version"],
                       "requires": distribution.get("requires") or []})
    wheels.append(supplement["additional_overlay_wheel"])
    if {item["name"] for item in wheels} != EXPECTED_DISTROS:
        raise RuntimeError("overlay lock does not describe the five approved distributions")
    verified_wheels = []
    for item in wheels:
        path = WHEELS / item["filename"]
        actual = sha256(path)
        if actual != item["sha256"]:
            raise RuntimeError(f"wheel SHA-256 mismatch for {path.name}: {actual}")
        with __import__("zipfile").ZipFile(path) as wheel:
            metadata_name = next(name for name in wheel.namelist() if name.endswith(".dist-info/METADATA"))
            metadata = BytesParser().parsebytes(wheel.read(metadata_name))
        if metadata["Name"].lower().replace("_", "-") != item["name"]:
            raise RuntimeError(f"wheel {path.name} embeds unexpected package name {metadata['Name']!r}")
        if metadata["Version"] != item["version"]:
            raise RuntimeError(f"wheel {path.name} embeds unexpected version {metadata['Version']!r}")
        expected_requires = item.get("requires") or item.get("requires_dist") or []
        actual_requires = metadata.get_all("Requires-Dist", [])
        if expected_requires != actual_requires:
            raise RuntimeError(f"wheel {path.name} dependency metadata differs from its approved lock")
        verified_wheels.append({**item, "path": str(path), "metadata_path": metadata_name,
                                "requires_python": metadata.get("Requires-Python"),
                                "requires_dist": actual_requires})

    if not OVERLAY.is_dir():
        raise RuntimeError(f"approved external overlay is missing: {OVERLAY}")
    installed = {}
    for metadata_path in sorted(OVERLAY.glob("*.dist-info/METADATA")):
        metadata = BytesParser().parsebytes(metadata_path.read_bytes())
        name = metadata["Name"].lower().replace("_", "-")
        installed[name] = {"version": metadata["Version"], "metadata_path": str(metadata_path),
                           "requires_dist": metadata.get_all("Requires-Dist", [])}
    if set(installed) != EXPECTED_DISTROS:
        raise RuntimeError(f"external overlay inventory differs from its lock: {sorted(installed)}")
    for item in verified_wheels:
        actual = installed[item["name"]]
        if actual["version"] != item["version"] or actual["requires_dist"] != item["requires_dist"]:
            raise RuntimeError(f"installed overlay metadata differs from the verified wheel for {item['name']}")
    for relpath in ("fla/__init__.py", "causal_conv1d/__init__.py"):
        if not (OVERLAY / relpath).is_file():
            raise RuntimeError(f"expected optional kernel source is absent from the overlay: {relpath}")
    return {"path": str(OVERLAY), "wheels": verified_wheels, "installed_distributions": installed,
            "optional_kernel_imports": "deferred to the exclusive GPU correctness process"}


def _historical_hashes(repo):
    return {name: sha256(repo / name) for name in HISTORICAL_FILES}


def _source_hashes(repo):
    return {name: sha256(repo / name) for name in SOURCE_FILES if (repo / name).is_file()}


def create_fixture(repo, fixture_dir, lock, hf_cache):
    os.environ.update(hf_cache["environment"])
    sys.path.insert(0, str(repo))
    experiment_dir = repo / "experiments/judge-manipulation"
    spec = importlib.util.spec_from_file_location("phase4_original_smoke", experiment_dir / "run_phase4_smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    info = module.build_fixture(lock, fixture_dir, count=32)
    if info["fixture_sha256"] != FIXTURE_SHA256:
        raise RuntimeError(f"recreated fixture SHA-256 differs from the approved plan: {info['fixture_sha256']}")
    required = {"fixture_path", "fixture_sha256", "record_count", "unique_source_records",
                "state_tokens_including_state_delimiter", "full_encoded_row_tokens", "branch_tokens",
                "question_id", "question_instructions"}
    missing = sorted(required - set(info))
    if missing:
        raise RuntimeError(f"fixture serialization is missing required report fields: {missing}")
    expected = {"record_count": 32, "state_tokens_including_state_delimiter": 2048,
                "full_encoded_row_tokens": 2099, "question_id": "judge_manipulation"}
    for name, value in expected.items():
        if info[name] != value:
            raise RuntimeError(f"fixture report field {name} differs from the registered shape: {info[name]!r}")
    if sha256(info["fixture_path"]) != FIXTURE_SHA256:
        raise RuntimeError("fixture file SHA-256 differs from its registered hash")
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--runtime-root", type=Path, default=Path.home() / ".cache/kev-phase4-shared/runs")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    started = time.monotonic()
    repo = args.repo.resolve()
    run_id = args.run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{5,63}", run_id):
        parser.error("--run-id must be 6 to 64 lowercase letters, digits, or hyphens")
    runtime_root = args.runtime_root.resolve()
    if runtime_root.as_posix().startswith("/mnt/"):
        parser.error("runtime data must be written under the Linux filesystem, not DrvFs")
    runtime_dir = runtime_root / run_id
    runtime_dir.mkdir(parents=True, exist_ok=False)
    fixture_dir = runtime_dir / "fixture"
    fixture_dir.mkdir()

    experiment_dir = repo / "experiments/judge-manipulation"
    plan_path = experiment_dir / "phase4-shared-memory-plan.json"
    model_lock_path = experiment_dir / "model-lock-phase-4.json"
    if sha256(plan_path) != PLAN_SHA256:
        raise RuntimeError("approved shared-memory plan bytes changed")
    if sha256(model_lock_path) != MODEL_LOCK_SHA256:
        raise RuntimeError("frozen Phase 4 model lock bytes changed")
    plan = read_json(plan_path)
    model_lock = read_json(model_lock_path)
    if model_lock.get("status") != "locked":
        raise RuntimeError("Phase 4 model lock is not locked")
    hf_cache = verify_local_model_cache(model_lock)
    os.environ.update(hf_cache["environment"])
    sampler_contract = verify_windows_sampler_contract(experiment_dir)
    historical = _historical_hashes(repo)
    freeze_expected = {
        "experiments/judge-manipulation/feasibility-gate-phase-4.json": plan["historical_failed_gate"]["sha256"],
        "experiments/judge-manipulation/AMENDMENT-shared-memory-phase-4.json": plan["amendment"]["sha256"],
    }
    for name, expected in freeze_expected.items():
        if historical[name] != expected:
            raise RuntimeError(f"frozen Phase 4 artifact changed: {name} has {historical[name]}, expected {expected}")
    overlay = overlay_inventory(experiment_dir / "phase4-shared-memory-overlay-lock.json",
                                experiment_dir / "phase4-shared-memory-overlay-supplement.json")
    primary_before = primary_environment(repo)
    dependency_files_before = {name: sha256(repo / name) for name in ("pyproject.toml", "uv.lock")}
    source_hashes = _source_hashes(repo)
    if len(source_hashes) != len(SOURCE_FILES):
        missing = sorted(set(SOURCE_FILES) - set(source_hashes))
        raise RuntimeError(f"not all benchmark source files exist yet: {missing}")

    fixture = create_fixture(repo, fixture_dir, model_lock, hf_cache)
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("fixture preparation unexpectedly initialized CUDA")
    result = {
        "schema_version": 1,
        "status": "cpu_preflight_passed",
        "run_id": run_id,
        "plan_sha256": PLAN_SHA256,
        "model_lock_sha256": MODEL_LOCK_SHA256,
        "repository": str(repo),
        "runtime_directory": str(runtime_dir),
        "primary_python": str(PRIMARY_PYTHON),
        "primary_environment_before": primary_before,
        "primary_dependency_files_before": dependency_files_before,
        "overlay": overlay,
        "huggingface_cache": hf_cache,
        "windows_sampler_contract": sampler_contract,
        "fixture": fixture,
        "historical_files_before": historical,
        "source_hashes_before": source_hashes,
        "setup_elapsed_seconds": round(time.monotonic() - started, 6),
        "gpu_context_initialized": False,
        "next_command": "phase4_shared_memory_runner.py --repo <repo> --run-dir <runtime_directory>",
    }
    result_path = runtime_dir / "setup.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "run_id": run_id, "runtime_directory": str(runtime_dir),
                      "plan_sha256": PLAN_SHA256, "model_lock_sha256": MODEL_LOCK_SHA256,
                      "fixture_sha256": fixture["fixture_sha256"], "setup_elapsed_seconds": result["setup_elapsed_seconds"],
                      "gpu_context_initialized": False}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
