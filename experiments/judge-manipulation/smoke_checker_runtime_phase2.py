#!/usr/bin/env python3
"""Run one bounded non-pilot JSON-format smoke against a local checker server."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
import subprocess
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


CONTEXT_SIZE = 8192
MAX_TOKENS = 32
SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "runtime_smoke", "strict": True, "schema": SCHEMA},
}
MESSAGES = [
    {
        "role": "system",
        "content": "Return exactly one JSON object matching the supplied schema.",
    },
    {"role": "user", "content": "Set ok to true."},
]


def request_json(url: str, *, payload: dict | None = None, timeout: float = 300) -> dict:
    body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"} if body is not None else {},
        method="POST" if body is not None else "GET",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"{url} failed: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def live_server_process(binary: Path, port: int) -> dict:
    expected = binary.resolve(strict=True)
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            arguments = (entry / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
            executable = (entry / "exe").resolve(strict=True)
        except (OSError, UnicodeDecodeError):
            continue
        if executable != expected or "--port" not in arguments:
            continue
        index = arguments.index("--port")
        if index + 1 < len(arguments) and arguments[index + 1] == str(port):
            return {"pid": int(entry.name), "executable": str(executable), "arguments": arguments}
    raise RuntimeError(f"no live llama-server process matches {expected} on port {port}")


def require_argument(arguments: list[str], flag: str, expected: str | None = None) -> None:
    if flag not in arguments:
        raise RuntimeError(f"live server command lacks required flag {flag}")
    if expected is not None:
        index = arguments.index(flag)
        if index + 1 >= len(arguments) or arguments[index + 1] != expected:
            raise RuntimeError(f"live server command {flag} value differs from {expected!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("qwen3-4b-q4km", "phi4-mini-q4km"))
    parser.add_argument("--base-url", default="http://127.0.0.1:18082/v1")
    parser.add_argument("--server-log", type=Path, required=True, help="captured output from the active model's server process")
    parser.add_argument("--report", type=Path, help="Optional report path outside Git, under the phase-2 cache")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    runtime = json.loads((here / "checker-runtime-lock-phase-2.json").read_text(encoding="utf-8"))
    server = runtime["server"]
    model = next((item for item in runtime["models"] if item["id"] == args.model), None)
    if model is None:
        raise RuntimeError(f"runtime lock has no model {args.model!r}")
    model_path = Path(model["gguf_path"])
    if not model_path.is_file() or sha256_file(model_path) != model["gguf_sha256"]:
        raise RuntimeError(f"local model bytes do not match the pinned runtime hash: {model_path}")
    if model.get("gguf_quantization") != "Q4_K_M":
        raise RuntimeError(f"runtime lock does not identify a Q4_K_M model: {args.model}")
    tokenizer_hashes = {}
    for item in model["tokenizer_files"]:
        path = Path(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"local tokenizer bytes do not match the pinned runtime hash: {path}")
        tokenizer_hashes[str(path)] = item["sha256"]
    source_root = Path.home() / ".cache" / "judge-manipulation-phase2" / "source" / "llama.cpp"
    source_commit = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if source_commit != runtime["runtime"]["source_commit"]:
        raise RuntimeError(f"llama.cpp source checkout differs from pinned commit: {source_commit}")
    binary = Path(server["binary"])
    binary_hash = sha256_file(binary)
    process = live_server_process(binary, int(server["port"]))
    arguments = process["arguments"]
    expected_arguments = (
        ("--model", str(model_path)),
        ("--alias", args.model),
        ("--host", "127.0.0.1"),
        ("--port", str(server["port"])),
        ("--ctx-size", str(model["context_size"])),
        ("--parallel", "1"),
        ("--batch-size", str(server["batch_size"])),
        ("--ubatch-size", str(server["ubatch_size"])),
        ("--gpu-layers", str(server["gpu_layers_argument"])),
        ("--flash-attn", "on"),
    )
    for flag, expected in expected_arguments:
        require_argument(arguments, flag, expected)
    require_argument(arguments, "--no-webui")
    require_argument(arguments, "--jinja")
    template_kwargs = "{\"enable_thinking\":false}" if args.model == "qwen3-4b-q4km" else "{}"
    require_argument(arguments, "--chat-template-kwargs", template_kwargs)

    base_url = args.base_url.rstrip("/")
    if base_url != server["base_url"].rstrip("/"):
        raise RuntimeError(f"smoke URL {base_url!r} differs from runtime lock {server['base_url']!r}")
    origin = base_url.removesuffix("/v1")
    is_qwen = args.model == "qwen3-4b-q4km"
    template_payload = {
        "model": args.model,
        "messages": MESSAGES,
        "add_generation_prompt": True,
        "response_format": RESPONSE_FORMAT,
    }
    completion_payload = {
        "model": args.model,
        "messages": MESSAGES,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "response_format": RESPONSE_FORMAT,
    }
    if is_qwen:
        template_payload["chat_template_kwargs"] = {"enable_thinking": False}
        completion_payload["chat_template_kwargs"] = {"enable_thinking": False}

    started = time.monotonic()
    health = request_json(f"{origin}/health")
    if health.get("status") != "ok":
        raise RuntimeError(f"server health was not ok: {health!r}")

    live_props = request_json(f"{origin}/props")
    live_alias = live_props.get("model_alias")
    live_model_path = live_props.get("model_path")
    generation = live_props.get("default_generation_settings")
    live_context = generation.get("n_ctx") if isinstance(generation, dict) else None
    if live_alias != args.model:
        raise RuntimeError(f"/props model_alias {live_alias!r} does not match {args.model!r}")
    if not isinstance(live_model_path, str) or not Path(live_model_path).is_absolute() or Path(live_model_path).resolve() != model_path.resolve():
        raise RuntimeError(f"/props model_path {live_model_path!r} does not match pinned {model_path}")
    if type(live_context) is not int or live_context != model["context_size"]:
        raise RuntimeError(f"/props default_generation_settings.n_ctx {live_context!r} does not match {model['context_size']}")
    if live_props.get("is_sleeping") is True:
        raise RuntimeError("/props reports that the loaded model is sleeping")

    model_list = request_json(f"{base_url}/models")
    ids = {entry.get("id") for entry in model_list.get("data", [])}
    if args.model not in ids:
        raise RuntimeError(f"server did not advertise requested model alias {args.model!r}: {sorted(ids)!r}")

    rendered = request_json(f"{origin}/apply-template", payload=template_payload)
    prompt = rendered.get("prompt")
    if not isinstance(prompt, str):
        raise RuntimeError(f"/apply-template returned no string prompt: {rendered!r}")
    tokenized = request_json(
        f"{origin}/tokenize",
        payload={
            "content": prompt,
            "add_special": True,
            "parse_special": True,
            "with_pieces": False,
        },
    )
    tokens = tokenized.get("tokens")
    if not isinstance(tokens, list):
        raise RuntimeError(f"/tokenize returned no token list: {tokenized!r}")
    prompt_tokens = len(tokens)
    if prompt_tokens + MAX_TOKENS > CONTEXT_SIZE:
        raise RuntimeError(
            f"smoke prompt exceeds context: {prompt_tokens} prompt tokens + {MAX_TOKENS} output tokens "
            f"> {CONTEXT_SIZE}"
        )

    completion = request_json(f"{base_url}/chat/completions", payload=completion_payload)
    try:
        content = completion["choices"][0]["message"]["content"]
        finish_reason = completion["choices"][0]["finish_reason"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"completion response has no first message: {completion!r}") from exc
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError(f"completion was not valid JSON: {content!r}") from exc
    if parsed != {"ok": True}:
        raise RuntimeError(f"completion did not satisfy smoke schema: {parsed!r}")
    if finish_reason != "stop":
        raise RuntimeError(f"completion ended with finish_reason={finish_reason!r}")

    server_log = args.server_log.read_text(encoding="utf-8", errors="replace")
    offload_lines = re.findall(r"[^\n]*offloaded\s+(\d+)/(\d+)\s+layers? to GPU[^\n]*", server_log, re.IGNORECASE)
    if not offload_lines:
        raise RuntimeError("server log has no observed GPU layer offload count")
    offloaded, total_layers = (int(value) for value in offload_lines[-1][:2])
    if offloaded != total_layers:
        raise RuntimeError(f"only {offloaded}/{total_layers} model layers were offloaded to GPU")

    usage = completion.get("usage", {})
    report = {
        "status": "passed",
        "model": args.model,
        "model_path": str(model_path.resolve()),
        "model_sha256": model["gguf_sha256"],
        "model_quantization": model["gguf_quantization"],
        "tokenizer_file_sha256": tokenizer_hashes,
        "base_url": base_url,
        "server_origin": origin,
        "context_size": CONTEXT_SIZE,
        "observed_context_size": live_context,
        "observed_props_model_alias": live_alias,
        "observed_props_model_path": str(Path(live_model_path).resolve()),
        "observed_v1_models_ids": sorted(ids),
        "server_binary_path": str(binary.resolve()),
        "server_binary_sha256": binary_hash,
        "llama_cpp_source_commit": source_commit,
        "server_process": process,
        "server_commandline_verified": True,
        "gpu_layers_argument": int(server["gpu_layers_argument"]),
        "gpu_layers_offloaded": offloaded,
        "gpu_layers_total": total_layers,
        "gpu_offload_log_lines": offload_lines,
        "server_log_path": str(args.server_log),
        "max_tokens": MAX_TOKENS,
        "prompt_tokens_from_server_tokenize": prompt_tokens,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "qwen_enable_thinking": False if is_qwen else None,
        "json_schema_response": parsed,
        "finish_reason": finish_reason,
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "pilot_prompts_used": False,
    }
    encoded = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding="utf-8")
    sys.stdout.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
