#!/usr/bin/env bash
# End-to-end MoE smoke: 12 training steps (3+3+3+3) + checkpoint load/generate smoke + HumanEval.
# Default model: Qwen/Qwen1.5-MoE-A2.7B (see configs/smoke_qwen3_moe.yaml).
# On GH200 with Qwen3-Coder-30B-A3B, export SMOKE_MOE_MODEL=Qwen/Qwen3-Coder-30B-A3B and
# replace model.name_or_path in configs/smoke_qwen3_moe/*.yaml or patch via sed below.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
if [[ -x "$PROJECT_DIR/.venv/bin/python3" ]]; then
  export PATH="$PROJECT_DIR/.venv/bin:$PATH"
fi

SMOKE_MODEL="${SMOKE_MOE_MODEL:-Qwen/Qwen1.5-MoE-A2.7B}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OPENCOMPOSER_EXIT_HARD="${OPENCOMPOSER_EXIT_HARD:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "=== OpenComposer MoE smoke (model=${SMOKE_MODEL}) ==="

python3 <<PY
# If SMOKE_MOE_MODEL is set, patch phase1 yaml base checkpoint id.
import os, pathlib
m = os.environ.get("SMOKE_MOE_MODEL")
if not m:
    raise SystemExit(0)
p = pathlib.Path("configs/smoke_qwen3_moe/cpt_phase1.yaml")
text = p.read_text().splitlines()
out = []
for line in text:
    s = line.strip()
    if s.startswith("name_or_path:") and "./checkpoints/" not in line:
        out.append(f'  name_or_path: "{m}"')
    else:
        out.append(line)
p.write_text("\n".join(out) + "\n")
PY

echo "[1/7] prepare_data (SFT)"
python3 scripts/prepare_data.py --stage sft --num_sft_examples 48

STEPS="${SMOKE_MAX_STEPS_PER_STAGE:-3}"
echo "[2/7] CPT phase1 (${STEPS} steps)"
python3 scripts/cpt_phase1.py --config configs/smoke_qwen3_moe/cpt_phase1.yaml --max_steps "${STEPS}"

echo "[3/7] CPT phase2 (${STEPS} steps)"
python3 scripts/cpt_phase2.py --config configs/smoke_qwen3_moe/cpt_phase2.yaml --max_steps "${STEPS}"

echo "[4/7] CPT phase3 SFT (${STEPS} steps)"
python3 scripts/cpt_phase3_sft.py --config configs/smoke_qwen3_moe/cpt_phase3_sft.yaml

echo "[5/7] Tool-use SFT (${STEPS} steps)"
python3 scripts/sft_tool_use.py --config configs/smoke_qwen3_moe/sft_toolu.yaml

echo "[6/7] Checkpoint smoke (load + short generate)"
export PRETRAIN_PATH="$(pwd)/checkpoints/smoke_qwen3_moe/stage2_sft"
python3 scripts/rl_smoke_fallback.py \
  --checkpoint_path "${PRETRAIN_PATH}" \
  --max_new_tokens 64 \
  --output ./evaluation_results/smoke/rl_fallback_smoke.json \
  --fallback_model "${SMOKE_MODEL}"

echo "[7/7] Eval (HumanEval subset)"
export OPENCOMPOSER_HUMANEVAL_LIMIT="${OPENCOMPOSER_HUMANEVAL_LIMIT:-8}"
python3 scripts/evaluate.py --checkpoint_path "${PRETRAIN_PATH}" --suite humaneval --output_dir ./evaluation_results/smoke

echo "=== Smoke complete. Checkpoints under checkpoints/smoke_qwen3_moe/ ==="
