#!/bin/bash
# Stage 3: RL with OpenRLHF — Dr. GRPO + k1 KL (Composer 2 §4), multi-turn agent, self-summary,
# nonlinear length / aux rewards via OPENCOMPOSER_* env vars in the Ray runtime.
#
# MoE router replay hooks live in opencomposer/moe (apply during HF forwards); vLLM rollouts do not
# yet export per-token expert indices — see README.
#
# Prerequisites:
#   bash scripts/install_openrlhf_local.sh
#   Stage 2 checkpoint at ./checkpoints/stage2_sft (or set PRETRAIN_PATH)
#
# Usage:
#   bash scripts/rl_training.sh
#
# One short cycle (tight batches) for smoke / CI:
#   export PRETRAIN_PATH=... SAVE_PATH=...
#   bash scripts/rl_one_cycle.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
if [[ -x "$PROJECT_DIR/.venv/bin/python3" ]]; then
    export PATH="$PROJECT_DIR/.venv/bin:$PATH"
fi

# Configuration (use absolute paths: Ray job workers don't share the submit CWD)
PRETRAIN_PATH="${PRETRAIN_PATH:-./checkpoints/stage2_sft}"
SAVE_PATH="${SAVE_PATH:-./checkpoints/stage3_rl}"
if [[ -d "$PRETRAIN_PATH" ]]; then
  PRETRAIN_PATH="$(realpath "$PRETRAIN_PATH")"
fi
SAVE_PATH="$(realpath -m "$SAVE_PATH")"
AGENT_FUNC_PATH="${PROJECT_DIR}/opencomposer/rl/agent_func.py"

# Optional smoke / tuning overrides (defaults match full run)
MAX_SAMPLES="${RL_MAX_SAMPLES:-10000}"
MAX_EPOCHS="${RL_MAX_EPOCHS:-1}"
MICRO_TRAIN_BATCH_SIZE="${RL_MICRO_TRAIN_BATCH_SIZE:-4}"
TRAIN_BATCH_SIZE="${RL_TRAIN_BATCH_SIZE:-64}"
MICRO_ROLLOUT_BATCH_SIZE="${RL_MICRO_ROLLOUT_BATCH_SIZE:-8}"
ROLLOUT_BATCH_SIZE="${RL_ROLLOUT_BATCH_SIZE:-64}"
PROMPT_MAX_LEN="${RL_PROMPT_MAX_LEN:-2048}"
GENERATE_MAX_LEN="${RL_GENERATE_MAX_LEN:-2048}"
SAVE_STEPS="${RL_SAVE_STEPS:-50}"
N_SAMPLES_PER_PROMPT="${RL_N_SAMPLES_PER_PROMPT:-4}"
# Colocated actor+vLLM on one GPU: vLLM sets requested_memory = total_vram * gpu_memory_utilization
# (fraction of *full* device memory). ~0.32 is too small for GLM-9B weights + activation profiling + KV.
# Use ~0.55–0.70 on 80GB+ cards; lower only if you reduce max_model_len / model size (override RL_VLLM_GPU_MEM_UTIL).
VLLM_GPU_MEM_UTIL="${RL_VLLM_GPU_MEM_UTIL:-0.62}"
INIT_KL_COEF="${RL_INIT_KL_COEF:-0.01}"
# vLLM/DeepSpeed sleep uses cumem remapping; on single-GPU colocate this often OOMs on wake_up.
RL_TRAINING_SLEEP="${RL_TRAINING_SLEEP:-0}"
# Eager mode avoids CUDA graph issues during colocated vLLM startup on some drivers/stacks.
RL_ENFORCE_EAGER="${RL_ENFORCE_EAGER:-1}"
# Engine core subprocess: RL_VLLM_MP=1 (default) matches upstream; set RL_VLLM_MP=0 for in-process worker.
RL_VLLM_MP="${RL_VLLM_MP:-1}"

# Verify prerequisites
if [[ ! -d "$PRETRAIN_PATH" ]] && [[ "$PRETRAIN_PATH" != *"/"* ]]; then
    echo "Error: Pretrained model not found at $PRETRAIN_PATH"
    echo "Run CPT phases + Stage 2 SFT first, or use a HF repo id:"
    echo "  export PRETRAIN_PATH=Qwen/Qwen3-Coder-30B-A3B"
    exit 1
fi

echo "=== OpenComposer Stage 3: RL Training ==="
echo "Pretrained model: $PRETRAIN_PATH"
echo "Save path: $SAVE_PATH"
echo "Agent func: $AGENT_FUNC_PATH"
echo "vLLM gpu_memory_utilization: $VLLM_GPU_MEM_UTIL (from RL_VLLM_GPU_MEM_UTIL)"
echo ""

# vLLM rollout only when vLLM imports and CUDA is available. PyPI vllm often installs
# cpu torch on aarch64; use HuggingFace generate via --vllm_num_engines 0 instead.
VLLM_ROLLOUT=1
if [[ -n "${FORCE_VLLM_ROLLOUT:-}" ]]; then
  VLLM_ROLLOUT="$FORCE_VLLM_ROLLOUT"
elif ! python3 -c "import vllm, torch; assert torch.version.cuda and torch.cuda.is_available()" 2>/dev/null; then
  VLLM_ROLLOUT=0
  echo "vLLM not used (missing vllm or no CUDA torch); rollouts use HF generate. Set FORCE_VLLM_ROLLOUT=1 to override."
fi

# Fallback mode when OpenRLHF is unavailable on this machine.
if ! python3 -c "import openrlhf" >/dev/null 2>&1; then
    echo "OpenRLHF is not installed; running fallback load+generate smoke instead."
    python3 scripts/rl_smoke_fallback.py \
        --checkpoint_path "$PRETRAIN_PATH" \
        --output "./evaluation_results/rl_fallback_smoke.json" \
        --max_new_tokens 64
    echo ""
    echo "=== Fallback RL smoke complete ==="
    echo "See: ./evaluation_results/rl_fallback_smoke.json"
    exit 0
fi

# Start or recover Ray (stale sessions yield "Can't find node_ip_address.json").
_ensure_ray_cluster() {
  if [[ "${RL_FORCE_RAY_RESTART:-0}" == "1" ]]; then
    echo "RL_FORCE_RAY_RESTART=1: stopping any existing Ray processes..."
    ray stop --force 2>/dev/null || true
    sleep 2
  fi
  local healthy=0
  if ray status &>/dev/null; then
    if python3 -c "import ray; ray.init(address='auto'); ray.shutdown()" 2>/dev/null; then
      healthy=1
    fi
  fi
  if [[ "$healthy" == "1" ]]; then
    echo "Using existing healthy Ray cluster."
    return 0
  fi
  echo "Starting Ray head (127.0.0.1, num-gpus=${RAY_NUM_GPUS:-1})..."
  ray stop --force 2>/dev/null || true
  sleep 2
  ray start --head --node-ip-address 127.0.0.1 --num-gpus "${RAY_NUM_GPUS:-1}" --disable-usage-stats
  sleep 3
  python3 -c "import ray; ray.init(address='auto'); ray.shutdown()" || {
    echo "ERROR: Ray failed to start. Try: ray stop --force && ray start --head --node-ip-address 127.0.0.1 --num-gpus 1"
    exit 1
  }
}

_ensure_ray_cluster

# Launch RL training via Ray
if [[ "$VLLM_ROLLOUT" == 1 ]]; then
  VLLM_ARGS=(
    --vllm_num_engines 1
    --vllm_tensor_parallel_size 1
    --vllm_gpu_memory_utilization "$VLLM_GPU_MEM_UTIL"
    --vllm_sync_backend nccl
  )
  if [[ "$RL_TRAINING_SLEEP" == "1" ]]; then
    VLLM_ARGS+=(--vllm_enable_sleep)
  fi
else
  VLLM_ARGS=(--vllm_num_engines 0)
fi

DEEPSPEED_SLEEP_ARGS=()
if [[ "$RL_TRAINING_SLEEP" == "1" ]]; then
  DEEPSPEED_SLEEP_ARGS=(--deepspeed_enable_sleep)
fi

ENFORCE_EAGER_ARGS=()
if [[ "$RL_ENFORCE_EAGER" == "1" ]]; then
  ENFORCE_EAGER_ARGS=(--enforce_eager)
fi

# Vendored vLLM aliases AsyncLLMEngine to V1 only; tune memory instead of VLLM_USE_V1 (noop here).
VLLM_MP_VAL="0"
if [[ "$RL_VLLM_MP" == "1" ]]; then
  VLLM_MP_VAL="1"
fi
RUNTIME_ENV_JSON="$(python3 -c "
import json
proj = r'''$PROJECT_DIR'''
pre = r'''$PRETRAIN_PATH'''
vmp = r'''$VLLM_MP_VAL'''
print(json.dumps({
  'working_dir': proj,
  'env_vars': {
    'PYTHONPATH': proj,
    'NCCL_P2P_DISABLE': '1',
    'NCCL_CUMEM_ENABLE': '0',
    'VLLM_ENABLE_V1_MULTIPROCESSING': vmp,
    'OPENCOMPOSER_MODEL_PATH': pre,
    'PRETRAIN_PATH': pre,
  },
}))
")"

RAY_WAIT_ARGS=()
if [[ "${RAY_JOB_NO_WAIT:-0}" == "1" ]]; then
  RAY_WAIT_ARGS=(--no-wait)
fi

ray job submit "${RAY_WAIT_ARGS[@]}" --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    --pretrain "$PRETRAIN_PATH" \
    --save_path "$SAVE_PATH" \
    --agent_func_path "$AGENT_FUNC_PATH" \
    --advantage_estimator dr_grpo \
    --kl_estimator k1 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 1 \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 1 \
    "${VLLM_ARGS[@]}" \
    "${ENFORCE_EAGER_ARGS[@]}" \
    --colocate_all_models \
    --init_kl_coef "$INIT_KL_COEF" \
    --actor_learning_rate 5e-7 \
    --micro_train_batch_size "$MICRO_TRAIN_BATCH_SIZE" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --micro_rollout_batch_size "$MICRO_ROLLOUT_BATCH_SIZE" \
    --rollout_batch_size "$ROLLOUT_BATCH_SIZE" \
    --n_samples_per_prompt "$N_SAMPLES_PER_PROMPT" \
    --max_samples "$MAX_SAMPLES" \
    --max_epochs "$MAX_EPOCHS" \
    --prompt_max_len "$PROMPT_MAX_LEN" \
    --generate_max_len "$GENERATE_MAX_LEN" \
    --zero_stage 2 \
    --param_dtype bf16 \
    --gradient_checkpointing \
    --save_steps "$SAVE_STEPS" \
    --logging_steps 1 \
    --adam_offload \
    --normalize_reward \
    --prompt_data OpenRLHF/prompt-collection-v0.1 \
    --input_key context_messages \
    --apply_chat_template \
    "${DEEPSPEED_SLEEP_ARGS[@]}"

echo ""
if [[ "${RAY_JOB_NO_WAIT:-0}" == "1" ]]; then
  echo "=== RL job submitted (RAY_JOB_NO_WAIT=1) ==="
  echo "Watch: ray job logs <job_id>   or    ray list jobs"
  echo "Checkpoints (on success): $SAVE_PATH"
else
  echo "=== RL Training Complete ==="
  echo "Checkpoints saved to: $SAVE_PATH"
fi
