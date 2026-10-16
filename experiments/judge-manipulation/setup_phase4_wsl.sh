#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd -- "$script_dir/../.." && pwd)"
uv_version=0.13.1
python_version=3.13.16
export JUDGE_MANIPULATION_PHASE4_HF_HOME="${JUDGE_MANIPULATION_PHASE4_HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface-phase4}"
export HF_HOME="$JUDGE_MANIPULATION_PHASE4_HF_HOME"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HUB_CACHE"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf "https://astral.sh/uv/${uv_version}/install.sh" | sh
  source "$HOME/.local/bin/env"
fi

read -r uv_name uv_actual_version _ < <(uv --version)
test "$uv_name" = uv
test "$uv_actual_version" = "$uv_version"
cd "$repo"
uv sync --locked --extra serve --python "$python_version" --link-mode copy
uv run --locked --python "$python_version" python -c 'import platform, torch, transformers, peft, huggingface_hub; print({"python": platform.python_version(), "torch": torch.__version__, "torch_cuda": torch.version.cuda, "transformers": transformers.__version__, "peft": peft.__version__, "huggingface_hub": huggingface_hub.__version__})'
