#!/usr/bin/env bash
# One short OpenRLHF PPO cycle (Dr. GRPO + k1, multi-turn agent) for end-to-end verification.
# Exports tight batch limits so a single training update completes quickly; then calls rl_training.sh.
#
# Required: openrlhf installed (bash scripts/install_openrlhf_local.sh), CUDA, Ray GPU.
#
# Usage:
#   export PRETRAIN_PATH=./checkpoints/pipeline_quick_verify/stage2_sft
#   export SAVE_PATH=./checkpoints/pipeline_quick_verify/stage3_rl
#   bash scripts/rl_one_cycle.sh
#
# Override any RL_* variable before calling (see scripts/rl_training.sh).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RL_MAX_SAMPLES="${RL_MAX_SAMPLES:-8}"
export RL_MAX_EPOCHS="${RL_MAX_EPOCHS:-1}"
export RL_TRAIN_BATCH_SIZE="${RL_TRAIN_BATCH_SIZE:-4}"
export RL_MICRO_TRAIN_BATCH_SIZE="${RL_MICRO_TRAIN_BATCH_SIZE:-1}"
export RL_ROLLOUT_BATCH_SIZE="${RL_ROLLOUT_BATCH_SIZE:-4}"
export RL_MICRO_ROLLOUT_BATCH_SIZE="${RL_MICRO_ROLLOUT_BATCH_SIZE:-2}"
export RL_N_SAMPLES_PER_PROMPT="${RL_N_SAMPLES_PER_PROMPT:-1}"
export RL_SAVE_STEPS="${RL_SAVE_STEPS:-1}"
export RL_PROMPT_MAX_LEN="${RL_PROMPT_MAX_LEN:-1024}"
export RL_GENERATE_MAX_LEN="${RL_GENERATE_MAX_LEN:-512}"
export RL_VLLM_GPU_MEM_UTIL="${RL_VLLM_GPU_MEM_UTIL:-0.35}"

# Restart Ray by default for short runs — avoids stale /tmp/ray sessions from older jobs.
export RL_FORCE_RAY_RESTART="${RL_FORCE_RAY_RESTART:-1}"

exec bash "$SCRIPT_DIR/rl_training.sh"
