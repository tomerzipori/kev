#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CACHE_ROOT="${PHASE2_RUNTIME_CACHE:-${HOME}/.cache/judge-manipulation-phase2}"
HF_HOME="${PHASE2_HF_HOME:-${HOME}/.cache/huggingface-phase2}"
HF_HUB_CACHE="${PHASE2_HF_HUB_CACHE:-${HF_HOME}/hub}"
HF_DATASETS_CACHE="${PHASE2_HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
RUNTIME_ROOT="${CACHE_ROOT}/runtime"
CUDART_LIB_DIR="$RUNTIME_ROOT/cudart/cudart-llama-b11425-bin-ubuntu-cuda-12.8-x64"
LLAMA_BIN_DIR="${RUNTIME_ROOT}/llama/llama-b11425"
SOURCE_ROOT="${CACHE_ROOT}/source/llama.cpp"
MODEL_ROOT="${CACHE_ROOT}/models"
CONVERTER_ROOT="${CACHE_ROOT}/converter-venv"
SERVER_PORT=18082

LLAMA_TAG=b11425
LLAMA_COMMIT=e117148a41d8e9bedb72e4c6c3f003ab0fe7f857
LLAMA_RELEASE=https://github.com/ggml-org/llama.cpp/releases/download/b11425
LLAMA_ASSET=llama-b11425-bin-ubuntu-cuda-12.8-x64.tar.gz
LLAMA_SHA256=ff7f7134ee3677bddf9ead356a1bc8b9cbc479d2b1e26b9e0d6753a5401e632b
CUDART_ASSET=cudart-llama-b11425-bin-ubuntu-cuda-12.8-x64.tar.gz
CUDART_SHA256=efe82ad6fea3820fef207e7cf73748760de3dcf604c1aaa9aa01d4c1ec2f79cb
GOMP_DEB=libgomp1_14.2.0-4ubuntu2~24.04.1_amd64.deb
GOMP_SHA256=e8a95ec58125b4933597f30ff56c2ae10edf90f287262e366d4b6edea3019144

QWEN_GGUF_REPO=Qwen/Qwen3-4B-GGUF
QWEN_GGUF_REV=bc640142c66e1fdd12af0bd68f40445458f3869b
QWEN_GGUF_FILE=Qwen3-4B-Q4_K_M.gguf
QWEN_GGUF_SHA256=7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5
QWEN_TOKENIZER_REPO=Qwen/Qwen3-4B
QWEN_TOKENIZER_REV=1cfa9a7208912126459214e8b04321603b3df60c
QWEN_TOKENIZER_JSON_SHA256=aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4
QWEN_TOKENIZER_CONFIG_SHA256=d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101
QWEN_TOKENIZER_TEMPLATE_SHA256=a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8

PHI_REPO=microsoft/Phi-4-mini-instruct
PHI_REV=cfbefacb99257ffa30c83adab238a50856ac3083
PHI_SHARD1_SHA256=bc703090b63eda16f639fa4de7ac54635c23105ab1da2f6ec4d3403151d38ee6
PHI_SHARD2_SHA256=7ff79b9d2d31076bac2663393451f6530f4fc8ca49b09002116c92c373dba983
PHI_TOKENIZER_JSON_SHA256=382cc235b56c725945e149cc25f191da667c836655efd0857b004320e90e91ea
PHI_TOKENIZER_CONFIG_SHA256=9c9b6bc0c94d95f69f826c41069a3e8b387ac3ced89601d201886e99240ac9db
PHI_TOKENIZER_TEMPLATE_SHA256=febf589225c9728ab791f52e8897d7607a823d45368f0a4c92fa68997b40cce9
PHI_F16_SHA256=8c61a6508269df3163fbc7baf2771567f42abfd0727cb113aeefaab5e52b03d2
PHI_Q4_SHA256=6ec1dc61cdda56cd97a4cc22f3aee71569fa60c989368616be92e8ad223fa6ab

export HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE
export QWEN_GGUF_REPO QWEN_GGUF_REV QWEN_TOKENIZER_REPO QWEN_TOKENIZER_REV PHI_REPO PHI_REV
mkdir -p "$CACHE_ROOT/downloads" "$RUNTIME_ROOT" "$MODEL_ROOT" "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }
check_sha() {
  local path="$1" expected="$2" actual
  [[ -f "$path" ]] || die "missing file: $path"
  actual="$(sha256_file "$path")"
  [[ "$actual" == "$expected" ]] || die "SHA-256 mismatch for $path: $actual (expected $expected)"
}

download_checked() {
  local url="$1" dest="$2" expected="$3" part
  if [[ -f "$dest" ]]; then
    if [[ "$(sha256_file "$dest")" == "$expected" ]]; then return 0; fi
    die "existing downloaded file has the wrong digest; inspect or remove explicitly: $dest"
  fi
  part="$dest.part"
  curl --fail --location --retry 4 --continue-at - "$url" -o "$part"
  check_sha "$part" "$expected"
  mv "$part" "$dest"
}

download_runtime() {
  local downloads="$CACHE_ROOT/downloads"
  local asset_url="$LLAMA_RELEASE/$LLAMA_ASSET"
  local cudart_url="$LLAMA_RELEASE/$CUDART_ASSET"

  download_checked "$asset_url" "$downloads/$LLAMA_ASSET" "$LLAMA_SHA256"
  download_checked "$cudart_url" "$downloads/$CUDART_ASSET" "$CUDART_SHA256"

  if [[ ! -x "$LLAMA_BIN_DIR/llama-server" ]]; then
    mkdir -p "$RUNTIME_ROOT/llama"
    tar -xzf "$downloads/$LLAMA_ASSET" -C "$RUNTIME_ROOT/llama" --strip-components=1
  fi
  if [[ ! -e "$RUNTIME_ROOT/cudart" ]]; then
    mkdir -p "$RUNTIME_ROOT/cudart"
    tar -xzf "$downloads/$CUDART_ASSET" -C "$RUNTIME_ROOT/cudart" --strip-components=1
  fi
  if [[ ! -f "$downloads/$GOMP_DEB" ]]; then
    (cd "$downloads" && apt-get download "libgomp1=14.2.0-4ubuntu2~24.04.1")
  fi
  check_sha "$downloads/$GOMP_DEB" "$GOMP_SHA256"
  if [[ ! -f "$RUNTIME_ROOT/deps/usr/lib/x86_64-linux-gnu/libgomp.so.1" ]]; then
    mkdir -p "$RUNTIME_ROOT/deps"
    dpkg-deb -x "$downloads/$GOMP_DEB" "$RUNTIME_ROOT/deps"
  fi

  local source_url="https://github.com/ggml-org/llama.cpp.git"
  if [[ ! -d "$SOURCE_ROOT/.git" ]]; then
    mkdir -p "$(dirname "$SOURCE_ROOT")"
    git clone --filter=blob:none --no-checkout "$source_url" "$SOURCE_ROOT"
  fi
  git -C "$SOURCE_ROOT" fetch --depth 1 origin "$LLAMA_COMMIT"
  git -C "$SOURCE_ROOT" checkout --detach "$LLAMA_COMMIT"
  [[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" == "$LLAMA_COMMIT" ]] || die 'llama.cpp source commit mismatch'

  printf 'llama.cpp tag=%s commit=%s\n' "$LLAMA_TAG" "$LLAMA_COMMIT"
  LD_LIBRARY_PATH="$CUDART_LIB_DIR:$RUNTIME_ROOT/deps/usr/lib/x86_64-linux-gnu:$LLAMA_BIN_DIR:${LD_LIBRARY_PATH:-}" \
    "$LLAMA_BIN_DIR/llama-server" --version
}

download_models() {
  local python="$ROOT/.venv/bin/python"
  [[ -x "$python" ]] || die "expected pre-existing Linux project Python at $python; do not recreate or modify it"
  "$python" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["QWEN_GGUF_REPO"],
    revision=os.environ["QWEN_GGUF_REV"],
    allow_patterns=["Qwen3-4B-Q4_K_M.gguf", "README.md", "LICENSE"],
)
snapshot_download(
    repo_id=os.environ["QWEN_TOKENIZER_REPO"],
    revision=os.environ["QWEN_TOKENIZER_REV"],
    allow_patterns=["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json", "config.json"],
)
snapshot_download(
    repo_id=os.environ["PHI_REPO"],
    revision=os.environ["PHI_REV"],
    allow_patterns=["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "model.safetensors.index.json", "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "LICENSE", "README.md"],
)
PY
}

find_snapshot_file() {
  local repo="$1" revision="$2" file="$3"
  local path="$HF_HUB_CACHE/models--${repo//\//--}/snapshots/$revision/$file"
  [[ -f "$path" ]] || die "missing pinned snapshot file: $path"
  printf '%s\n' "$path"
}

setup_converter() {
  local uv="${UV_BIN:-$HOME/.local/bin/uv}"
  [[ -x "$uv" ]] || die "uv 0.13.1 expected at $uv"
  [[ "$($uv --version | awk '{print $2}')" == '0.13.1' ]] || die "expected uv 0.13.1, found $($uv --version)"
  if [[ ! -x "$CONVERTER_ROOT/bin/python" ]]; then "$uv" venv --python 3.13.16 "$CONVERTER_ROOT"; fi
  [[ "$("$CONVERTER_ROOT/bin/python" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')" == 3.13.16 ]] || die 'converter Python must be 3.13.16'
  "$uv" pip sync --python "$CONVERTER_ROOT/bin/python" \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match \
    "$ROOT/experiments/judge-manipulation/converter-requirements-phase-2.lock.txt"
  "$uv" pip freeze --python "$CONVERTER_ROOT/bin/python" > "$CACHE_ROOT/converter-requirements.lock.txt"
  printf 'converter env: %s\n' "$CONVERTER_ROOT"
  "$CONVERTER_ROOT/bin/python" - <<'PY'
import sys, torch, transformers, sentencepiece, numpy
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("sentencepiece", sentencepiece.__version__)
print("numpy", numpy.__version__)
PY
  cat "$CACHE_ROOT/converter-requirements.lock.txt"
}

convert_phi() {
  [[ -x "$CONVERTER_ROOT/bin/python" ]] || die "create pinned converter env first: $0 converter-env"
  local snapshot="$HF_HUB_CACHE/models--microsoft--Phi-4-mini-instruct/snapshots/$PHI_REV"
  local shard1="$snapshot/model-00001-of-00002.safetensors"
  local shard2="$snapshot/model-00002-of-00002.safetensors"
  local f16="$MODEL_ROOT/phi-4-mini-instruct-f16.gguf"
  local q4="$MODEL_ROOT/phi-4-mini-instruct-Q4_K_M.gguf"
  local f16_part="$MODEL_ROOT/.phi-4-mini-instruct-f16.partial.$$"
  local q4_part="$MODEL_ROOT/.phi-4-mini-instruct-Q4_K_M.partial.$$"
  [[ -f "$shard1" && -f "$shard2" ]] || die 'Phi source shards are not fully downloaded'
  check_sha "$shard1" "$PHI_SHARD1_SHA256"
  check_sha "$shard2" "$PHI_SHARD2_SHA256"
  if [[ ! -f "$f16" ]]; then
    [[ ! -e "$f16_part" ]] || die "conversion attempt path already exists: $f16_part"
    env CUDA_VISIBLE_DEVICES='' "$CONVERTER_ROOT/bin/python" "$SOURCE_ROOT/convert_hf_to_gguf.py" "$snapshot" --outfile "$f16_part" --outtype f16 --use-temp-file
    [[ -s "$f16_part" ]] || die 'Phi F16 converter wrote no data'
    mv "$f16_part" "$f16"
  fi
  [[ -f "$f16" ]] || die 'Phi F16 conversion did not produce an output file'
  check_sha "$f16" "$PHI_F16_SHA256"
  if [[ ! -f "$q4" ]]; then
    [[ ! -e "$q4_part" ]] || die "quantization attempt path already exists: $q4_part"
    env CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH="$CUDART_LIB_DIR:$RUNTIME_ROOT/deps/usr/lib/x86_64-linux-gnu:$LLAMA_BIN_DIR:${LD_LIBRARY_PATH:-}" \
      "$LLAMA_BIN_DIR/llama-quantize" "$f16" "$q4_part" Q4_K_M
    [[ -s "$q4_part" ]] || die 'Phi quantizer wrote no data'
    mv "$q4_part" "$q4"
  fi
  [[ -f "$q4" ]] || die 'Phi Q4_K_M quantization did not produce an output file'
  check_sha "$q4" "$PHI_Q4_SHA256"
  printf 'Phi F16: %s %s bytes\n' "$(sha256_file "$f16")" "$(stat -c %s "$f16")"
  printf 'Phi Q4_K_M: %s %s bytes\n' "$(sha256_file "$q4")" "$(stat -c %s "$q4")"
}

verify_models() {
  local qwen
  qwen="$(find_snapshot_file "$QWEN_GGUF_REPO" "$QWEN_GGUF_REV" "$QWEN_GGUF_FILE")"
  check_sha "$qwen" "$QWEN_GGUF_SHA256"
  local qwen_tokenizer="$HF_HUB_CACHE/models--Qwen--Qwen3-4B/snapshots/$QWEN_TOKENIZER_REV"
  check_sha "$qwen_tokenizer/tokenizer.json" "$QWEN_TOKENIZER_JSON_SHA256"
  check_sha "$qwen_tokenizer/tokenizer_config.json" "$QWEN_TOKENIZER_CONFIG_SHA256"
  local phi="$MODEL_ROOT/phi-4-mini-instruct-Q4_K_M.gguf"
  [[ -f "$phi" ]] || die 'Phi converted Q4_K_M model is missing'
  check_sha "$phi" "$PHI_Q4_SHA256"
  check_sha "$MODEL_ROOT/phi-4-mini-instruct-f16.gguf" "$PHI_F16_SHA256"
  local phi_tokenizer="$HF_HUB_CACHE/models--microsoft--Phi-4-mini-instruct/snapshots/$PHI_REV"
  check_sha "$phi_tokenizer/tokenizer.json" "$PHI_TOKENIZER_JSON_SHA256"
  check_sha "$phi_tokenizer/tokenizer_config.json" "$PHI_TOKENIZER_CONFIG_SHA256"
  printf 'qwen3-4b-q4km %s %s bytes\n' "$(sha256_file "$qwen")" "$(stat -Lc %s "$qwen")"
  printf 'phi4-mini-q4km %s %s bytes\n' "$(sha256_file "$phi")" "$(stat -Lc %s "$phi")"
}

serve() {
  local checker="${1:-}"
  local model alias kwargs
  case "$checker" in
    qwen3-4b-q4km)
      model="$(find_snapshot_file "$QWEN_GGUF_REPO" "$QWEN_GGUF_REV" "$QWEN_GGUF_FILE")"
      check_sha "$model" "$QWEN_GGUF_SHA256"
      local qwen_tokenizer="$HF_HUB_CACHE/models--Qwen--Qwen3-4B/snapshots/$QWEN_TOKENIZER_REV"
      check_sha "$qwen_tokenizer/tokenizer.json" "$QWEN_TOKENIZER_JSON_SHA256"
      check_sha "$qwen_tokenizer/tokenizer_config.json" "$QWEN_TOKENIZER_CONFIG_SHA256"
      alias=qwen3-4b-q4km
      kwargs='{"enable_thinking":false}'
      ;;
    phi4-mini-q4km)
      model="$MODEL_ROOT/phi-4-mini-instruct-Q4_K_M.gguf"
      [[ -f "$model" ]] || die "missing $model; run convert-phi first"
      check_sha "$model" "$PHI_Q4_SHA256"
      local phi_tokenizer="$HF_HUB_CACHE/models--microsoft--Phi-4-mini-instruct/snapshots/$PHI_REV"
      check_sha "$phi_tokenizer/tokenizer.json" "$PHI_TOKENIZER_JSON_SHA256"
      check_sha "$phi_tokenizer/tokenizer_config.json" "$PHI_TOKENIZER_CONFIG_SHA256"
      alias=phi4-mini-q4km
      kwargs='{}'
      ;;
    *) die 'usage: setup_checker_runtime_phase2.sh serve-qwen | serve-phi' ;;
  esac
  [[ -x "$LLAMA_BIN_DIR/llama-server" ]] || die "missing llama-server; run runtime first"
  exec 9>"$CACHE_ROOT/llama-server.lock"
  flock -n 9 || die 'another checker server launched by this script holds the runtime lock'
  if curl --silent --fail "http://127.0.0.1:$SERVER_PORT/health" >/dev/null; then
    die "a server already responds on 127.0.0.1:$SERVER_PORT"
  fi
  exec env LD_LIBRARY_PATH="$CUDART_LIB_DIR:$RUNTIME_ROOT/deps/usr/lib/x86_64-linux-gnu:$LLAMA_BIN_DIR:${LD_LIBRARY_PATH:-}" \
    "$LLAMA_BIN_DIR/llama-server" \
    --model "$model" --alias "$alias" \
    --host 127.0.0.1 --port "$SERVER_PORT" --no-webui -lv 4 \
    --ctx-size 8192 --parallel 1 --batch-size 512 --ubatch-size 128 \
    --gpu-layers 99 --flash-attn on --jinja \
    --chat-template-kwargs "$kwargs"
}

case "${1:-}" in
  runtime) download_runtime ;;
  models) download_models ;;
  converter-env) setup_converter ;;
  convert-phi) convert_phi ;;
  verify) verify_models ;;
  serve-qwen) serve qwen3-4b-q4km ;;
  serve-phi) serve phi4-mini-q4km ;;
  *)
    cat <<USAGE
Usage: $0 {runtime|models|converter-env|convert-phi|verify|serve-qwen|serve-phi}

Run under WSL Ubuntu 24.04 from the Windows-mounted repository. All weights,
build/runtime files, and the isolated conversion environment live outside Git.
USAGE
    exit 2
    ;;
esac
