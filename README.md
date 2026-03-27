# OpenComposer

A miniature end-to-end reproduction of the Composer 2 training pipeline on a single NVIDIA GH200 480GB, using GLM-4-9B as the base model.

## Pipeline Overview

| Stage | Description | Script |
|-------|-------------|--------|
| 0 | Data preparation (code corpus, SFT data, RL environments) | `scripts/prepare_data.py` |
| 1 | Continued pretraining on code | `scripts/continued_pretraining.py` |
| 2 | SFT for tool-use format | `scripts/sft_tool_use.py` |
| 3 | RL training with tool use + self-summarization | `scripts/rl_training.sh` |
| 4 | Evaluation (HumanEval, SWE-bench Lite) | `scripts/evaluate.py` |

## Quick Start

```bash
pip install -r requirements.txt

# Stage 0: Prepare data
python scripts/prepare_data.py --stage all

# Stage 1: Continued pretraining
deepspeed scripts/continued_pretraining.py --deepspeed configs/deepspeed_zero2.json

# Stage 2: SFT
deepspeed scripts/sft_tool_use.py --deepspeed configs/deepspeed_zero2.json

# Stage 3: RL training
bash scripts/rl_training.sh

# Stage 4: Evaluate
python scripts/evaluate.py --checkpoint_path <path>
```

## Hardware Requirements

- 1x NVIDIA GH200 480GB (96GB HBM3 + 480GB LPDDR5X)
- Docker for sandboxed RL environments

## Architecture

```
GLM-4-9B (base)
    |
    v
Continued Pretraining (code corpus, causal LM)
    |
    v
SFT (tool-use format, multi-turn conversations)
    |
    v
RL + Self-Summarization (REINFORCE++-baseline in sandboxed coding envs)
    |
    v
Evaluation (HumanEval, MBPP, SWE-bench Lite)
```
