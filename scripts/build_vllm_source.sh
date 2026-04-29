#!/usr/bin/env bash
# Build vLLM from source against the CUDA PyTorch already in .venv (torch 2.11+cu128).
#
# vLLM's requirements/cuda.txt pins torch==2.9.0; a normal `pip install -e .` replaces
# your GPU torch with 2.9 (often breaking flash-attn and forcing a CPU/ACL build on ARM).
# We use `--no-deps` for the editable install, then add CUDA-only extras without torch.
#
# Usage (repo root):
#   bash scripts/build_vllm_source.sh
# Optional: VLLM_TAG=v0.13.0  TORCH_CUDA_ARCH_LIST=9.0  MAX_JOBS=8

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
TAG="${VLLM_TAG:-v0.13.0}"
SRC="${VLLM_SRC:-$ROOT/.vendor/vllm}"

export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export MAX_JOBS="${MAX_JOBS:-8}"
export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-cuda}"

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "Missing $ROOT/.venv; create venv first." >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$ROOT/.venv/bin/activate"

mkdir -p "$ROOT/.vendor"
if [[ ! -d "$SRC/.git" ]]; then
  git clone --depth 1 --branch "$TAG" https://github.com/vllm-project/vllm.git "$SRC"
else
  git -C "$SRC" fetch --depth 1 origin "refs/tags/$TAG:refs/tags/$TAG" 2>/dev/null || true
  git -C "$SRC" checkout -q "$TAG"
fi

echo "Pinning PyTorch CUDA stack (do not use vLLM's torch==2.9 pin)..."
uv pip install "torch==2.11.0" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

uv pip install 'cmake>=3.26' ninja 'packaging>=24.2' 'setuptools>=77.0.3,<81' setuptools-scm wheel jinja2

# Ubuntu split toolkit: nvcc lives under /usr/local/cuda but headers are in /usr/include.
# PyTorch CMake looks for cuda_runtime.h under ${CUDA_HOME}/include — symlink if missing.
if [[ ! -f "${CUDA_HOME:-/usr/local/cuda}/include/cuda_runtime.h" ]]; then
  echo "CUDA headers not under \$CUDA_HOME/include; adding symlinks from /usr/include (needs sudo)..."
  sudo sh -c 'cd /usr/local/cuda/include && for f in /usr/include/cuda* /usr/include/cublas* /usr/include/cufile* /usr/include/cufft* /usr/include/curand* /usr/include/cusolver* /usr/include/cusparse* /usr/include/cuuint* /usr/include/builtin* /usr/include/nvToolsExt* /usr/include/nvrtc* /usr/include/sm_* /usr/include/thrust; do [ -e "$f" ] && ln -sfn "$f" . ; done'
fi

echo "Building vLLM $TAG (CUDA_HOME=$CUDA_HOME arch=$TORCH_CUDA_ARCH_LIST jobs=$MAX_JOBS)"
cd "$SRC"
rm -rf build .deps

uv pip install -e . --no-build-isolation --no-deps

echo "Installing vLLM CUDA extras (no torch lines from cuda.txt)..."
uv pip install numba==0.61.2 'ray[cgraph]>=2.48.0' flashinfer-python==0.5.3

python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, torch.version.cuda)"
