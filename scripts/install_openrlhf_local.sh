#!/usr/bin/env bash
# Editable OpenRLHF install for Linux aarch64 (and x86_64): avoids broken PyPI wheels
# and keeps CUDA PyTorch (install torch+cu* separately; do not use openrlhf[vllm] on
# aarch64 unless you have a vLLM build that matches your torch — PyPI vllm may replace
# torch with a CPU build).
#
# Usage (from repo root, venv active):
#   bash scripts/install_openrlhf_local.sh
#
# Optional: OPENRLHF_TAG=v0.9.2 (default), OPENRLHF_DIR=path to existing clone

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TAG="${OPENRLHF_TAG:-v0.9.2}"
VENDOR="${OPENRLHF_DIR:-"$PROJECT_DIR/.vendor/OpenRLHF"}"

mkdir -p "$(dirname "$VENDOR")"
if [[ ! -d "$VENDOR/.git" ]]; then
  git clone --depth 1 --branch "$TAG" https://github.com/OpenRLHF/OpenRLHF.git "$VENDOR"
else
  git -C "$VENDOR" fetch --depth 1 origin "refs/tags/$TAG:refs/tags/$TAG" 2>/dev/null || true
  git -C "$VENDOR" checkout -q "$TAG"
fi

SETUP="$VENDOR/setup.py"
if ! grep -q "manylinux_2_28_aarch64" "$SETUP"; then
  python3 <<PY
from pathlib import Path
p = Path("$SETUP")
text = p.read_text()
old = """        if platform.system() == \"Linux\":
            platform_tag = \"manylinux1_x86_64\"
"""
new = """        if platform.system() == \"Linux\":
            machine = platform.machine().lower()
            if machine in (\"aarch64\", \"arm64\"):
                platform_tag = \"manylinux_2_28_aarch64\"
            else:
                platform_tag = \"manylinux1_x86_64\"
"""
if old not in text:
    raise SystemExit("setup.py layout changed; patch OpenRLHF/get_tag manually")
p.write_text(text.replace(old, new, 1))
PY
  echo "Patched $SETUP for Linux aarch64 wheel tags."
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; install https://github.com/astral-sh/uv or: pip install uv"
  exit 1
fi

uv pip install -e "$VENDOR"
echo "openrlhf installed from $VENDOR ($TAG). flash-attn should already match OpenRLHF requirements.txt."
echo ""
echo "PPO/RL: OpenRLHF links against vLLM. Prebuilt vLLM wheels may NOT match your PyTorch ABI (especially aarch64)."
echo "If 'import vllm' fails with undefined symbols, build vLLM from source against this torch, or use HF-only stages"
echo "(e.g. python scripts/sft_tool_use.py --config ...)."
