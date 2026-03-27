#!/bin/bash
# Stage 3: RL training with OpenRLHF (REINFORCE++-baseline with multi-turn agent)
#
# This script launches the full RL training pipeline with:
# - Multi-turn agent execution (tool use in sandboxed environments)
# - REINFORCE++-baseline (no critic needed, lower memory)
# - Self-summarization integrated into the agent loop
# - vLLM for fast generation
#
# Prerequisites:
#   pip install openrlhf[vllm]
#   Stage 2 checkpoint must exist at ./checkpoints/stage2_sft
#
# Usage:
#   bash scripts/rl_training.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Configuration
PRETRAIN_PATH="${PRETRAIN_PATH:-./checkpoints/stage2_sft}"
SAVE_PATH="${SAVE_PATH:-./checkpoints/stage3_rl}"
AGENT_FUNC_PATH="${PROJECT_DIR}/opencomposer/rl/agent_func.py"

# Verify prerequisites
if [ ! -d "$PRETRAIN_PATH" ]; then
    echo "Error: Pretrained model not found at $PRETRAIN_PATH"
    echo "Run Stage 1 (continued pretraining) and Stage 2 (SFT) first."
    echo ""
    echo "For testing, you can use the base model directly:"
    echo "  export PRETRAIN_PATH=THUDM/glm-4-9b-chat"
    exit 1
fi

echo "=== OpenComposer Stage 3: RL Training ==="
echo "Pretrained model: $PRETRAIN_PATH"
echo "Save path: $SAVE_PATH"
echo "Agent func: $AGENT_FUNC_PATH"
echo ""

# Start Ray head node (if not already running)
if ! ray status &>/dev/null; then
    echo "Starting Ray head node..."
    ray start --head --node-ip-address 0.0.0.0 --num-gpus 1
    sleep 2
fi

# Launch RL training via Ray
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="{\"working_dir\": \"${PROJECT_DIR}\", \"env_vars\": {\"PYTHONPATH\": \"${PROJECT_DIR}\"}}" \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    --pretrain "$PRETRAIN_PATH" \
    --save_path "$SAVE_PATH" \
    --agent_func_path "$AGENT_FUNC_PATH" \
    --advantage_estimator reinforce_baseline \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 1 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.4 \
    --colocate_all_models \
    --init_kl_coef 0.01 \
    --actor_learning_rate 5e-7 \
    --micro_train_batch_size 4 \
    --train_batch_size 64 \
    --micro_rollout_batch_size 8 \
    --rollout_batch_size 64 \
    --max_samples 10000 \
    --max_epochs 1 \
    --prompt_max_len 2048 \
    --generate_max_len 2048 \
    --zero_stage 2 \
    --bf16 \
    --gradient_checkpointing \
    --save_steps 50 \
    --logging_steps 1 \
    --adam_offload \
    --normalize_reward \
    --prompt_data OpenRLHF/prompt-collection-v0.1 \
    --input_key context_messages \
    --apply_chat_template \
    --vllm_sync_backend gloo \
    --vllm_enable_sleep \
    --deepspeed_enable_sleep

echo ""
echo "=== RL Training Complete ==="
echo "Checkpoints saved to: $SAVE_PATH"
