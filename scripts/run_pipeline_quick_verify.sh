#!/usr/bin/env bash
# Stages: data → CPT×2 → CPT-SFT → SFT → MTP → OpenRLHF (one RL cycle) → benchmark compare
# (SFT vs RL). Uses Qwen2.5-0.5B and short training by default. CPU-safe dtypes when no CUDA.
# Real RL needs CUDA + OpenRLHF (`bash scripts/install_openrlhf_local.sh`).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"
if [[ -x "$ROOT/.venv/bin/python3" ]]; then PATH="$ROOT/.venv/bin:$PATH"; fi

PY="${PYTHON:-python3}"
STEPS="${PIPELINE_VERIFY_STEPS:-2}"
VERIFY_MODEL="${PIPELINE_VERIFY_BASE_MODEL:-Qwen/Qwen2.5-0.5B}"

echo "=== Pipeline quick verify (steps/stage=$STEPS, base model=$VERIFY_MODEL) ==="

export OPENCOMPOSER_EXIT_HARD="${OPENCOMPOSER_EXIT_HARD:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"


# Pin the small base model in verify configs (only the first hub id line in phase1).
$PY <<PY
import pathlib
root = pathlib.Path(".")
model = "${VERIFY_MODEL}"
for name in ["cpt_phase1.yaml"]:
    p = root / "configs/pipeline_quick_verify" / name
    if not p.exists():
        continue
    lines = p.read_text().splitlines()
    out = []
    for line in lines:
        if line.strip().startswith("name_or_path:") and "/checkpoints/" not in line:
            out.append(f'  name_or_path: "{model}"')
        else:
            out.append(line)
    p.write_text("\n".join(out) + "\n")
PY

echo "[1/10] prepare_data"
$PY scripts/prepare_data.py --stage sft --num_sft_examples 32

echo "[2/10] CPT phase 1"
$PY scripts/cpt_phase1.py --config configs/pipeline_quick_verify/cpt_phase1.yaml --max_steps "$STEPS"

echo "[3/10] CPT phase 2"
$PY scripts/cpt_phase2.py --config configs/pipeline_quick_verify/cpt_phase2.yaml --max_steps "$STEPS"

echo "[4/10] CPT phase 3 (SFT-style)"
$PY scripts/cpt_phase3_sft.py --config configs/pipeline_quick_verify/cpt_phase3_sft.yaml

echo "[5/10] Tool-use SFT"
$PY scripts/sft_tool_use.py --config configs/pipeline_quick_verify/sft_toolu.yaml

echo "[6/10] MTP self-distill (1 outer step in yaml)"
$PY scripts/train_mtp.py --config configs/pipeline_quick_verify/mtp.yaml

SFT_CKPT="$ROOT/checkpoints/pipeline_quick_verify/stage2_sft"
RL_CKPT="$ROOT/checkpoints/pipeline_quick_verify/stage3_rl"

echo "[7/10] Stage 3: OpenRLHF — one RL training cycle (Dr. GRPO + k1, agent_func)"
export PRETRAIN_PATH="$SFT_CKPT"
export SAVE_PATH="$RL_CKPT"
if ! $PY -c "import openrlhf" 2>/dev/null; then
  echo "ERROR: OpenRLHF is not installed. Install with: bash scripts/install_openrlhf_local.sh"
  exit 1
fi
if ! $PY -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "ERROR: CUDA is required for OpenRLHF + vLLM/colocated training in this script."
  exit 1
fi
bash "$SCRIPT_DIR/rl_one_cycle.sh"

echo "[8/10] RL checkpoint smoke (load + short generate)"
$PY scripts/rl_smoke_fallback.py \
  --checkpoint_path "$RL_CKPT" \
  --max_new_tokens 64 \
  --output "$ROOT/evaluation_results/quick_verify/rl_post_training_smoke.json" \
  --fallback_model "$VERIFY_MODEL" || echo "WARN: rl_smoke_fallback failed (non-fatal)"

export OPENCOMPOSER_HUMANEVAL_LIMIT="${OPENCOMPOSER_HUMANEVAL_LIMIT:-3}"

echo "[9/10] Benchmark compare: SFT baseline (pre-RL) vs RL checkpoint (HumanEval)"
$PY scripts/evaluate.py \
  --head_to_head "$SFT_CKPT" "$RL_CKPT" \
  --suite "${PIPELINE_EVAL_SUITE:-humaneval}" \
  --output_dir "$ROOT/evaluation_results/quick_verify" \
  --head_labels sft_baseline after_one_rl_cycle

if [[ "${PIPELINE_COMPARE_HUB:-0}" == "1" ]]; then
  echo "[10/10] Extra compare: raw HF hub id vs RL checkpoint"
  $PY scripts/evaluate.py \
    --head_to_head "$VERIFY_MODEL" "$RL_CKPT" \
    --suite "${PIPELINE_EVAL_SUITE:-humaneval}" \
    --output_dir "$ROOT/evaluation_results/quick_verify" \
    --head_labels hub_base after_one_rl_cycle
else
  echo "[10/10] Skip hub vs RL (set PIPELINE_COMPARE_HUB=1 to run; saves eval time)"
fi

echo "Done. Checkpoints: pipeline_quick_verify/ (stage3_rl = RL output). Results: evaluation_results/quick_verify/"
echo "=== Full pipeline (incl. one RL cycle) succeeded ==="
