#!/usr/bin/env python3
"""Stage 0: Data preparation for the OpenComposer pipeline.

Usage:
    python scripts/prepare_data.py --stage all
    python scripts/prepare_data.py --stage sft --num_sft_examples 10000
    python scripts/prepare_data.py --stage rl_tasks
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opencomposer.data.sft_dataset import save_sft_dataset, TOOL_SCHEMAS
from opencomposer.data.rl_prompts import load_rl_tasks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def prepare_sft_data(output_dir: str, num_examples: int, seed: int):
    """Generate and save the synthetic SFT tool-use dataset."""
    logger.info("=== Preparing SFT tool-use dataset ===")
    save_sft_dataset(output_dir, num_examples=num_examples, seed=seed)
    logger.info("SFT dataset saved to %s", output_dir)


def prepare_rl_tasks(output_dir: str, source: str):
    """Load and save RL task definitions."""
    logger.info("=== Preparing RL task definitions ===")
    tasks = load_rl_tasks(source=source)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    task_dicts = []
    for t in tasks:
        task_dicts.append({
            "task_id": t.task_id,
            "description": t.description,
            "repo": t.repo,
            "base_commit": t.base_commit,
            "test_command": t.test_command,
            "setup_commands": t.setup_commands,
            "fail_to_pass": t.fail_to_pass,
            "pass_to_pass": t.pass_to_pass,
            "metadata": t.metadata,
        })

    with open(out_path / "tasks.json", "w") as f:
        json.dump(task_dicts, f, indent=2)

    logger.info("Saved %d RL tasks to %s/tasks.json", len(task_dicts), output_dir)


def prepare_tool_schemas(output_dir: str):
    """Save the tool schema definitions used across the pipeline."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "tool_schemas.json", "w") as f:
        json.dump(TOOL_SCHEMAS, f, indent=2)
    logger.info("Tool schemas saved to %s/tool_schemas.json", output_dir)


def verify_code_corpus():
    """Verify that the code corpus can be accessed (streaming check)."""
    logger.info("=== Verifying code corpus access ===")
    try:
        from datasets import load_dataset
        ds = load_dataset("bigcode/starcoderdata", data_dir="python", split="train", streaming=True)
        sample = next(iter(ds))
        logger.info("Code corpus accessible. Sample keys: %s, content length: %d",
                     list(sample.keys()), len(sample.get("content", "")))
    except Exception as e:
        logger.warning("Could not access code corpus: %s", e)
        logger.warning("You may need to accept the dataset license on HuggingFace.")
        logger.warning("Continued pretraining (Stage 1) will need this dataset.")


def main():
    parser = argparse.ArgumentParser(description="Prepare data for OpenComposer pipeline")
    parser.add_argument("--stage", choices=["sft", "rl_tasks", "verify_corpus", "all"], default="all")
    parser.add_argument("--sft_output_dir", default="./data/sft_tool_use")
    parser.add_argument("--rl_output_dir", default="./data/rl_tasks")
    parser.add_argument("--schema_output_dir", default="./data")
    parser.add_argument("--num_sft_examples", type=int, default=5000)
    parser.add_argument("--rl_source", choices=["builtin", "swebench"], default="builtin")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_tool_schemas(args.schema_output_dir)

    if args.stage in ("sft", "all"):
        prepare_sft_data(args.sft_output_dir, args.num_sft_examples, args.seed)

    if args.stage in ("rl_tasks", "all"):
        prepare_rl_tasks(args.rl_output_dir, args.rl_source)

    if args.stage in ("verify_corpus", "all"):
        verify_code_corpus()

    logger.info("=== Data preparation complete ===")


if __name__ == "__main__":
    main()
