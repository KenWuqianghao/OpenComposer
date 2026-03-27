#!/usr/bin/env python3
"""Stage 4: Unified evaluation script for all pipeline stages.

Runs HumanEval and agent-based coding evaluations at each checkpoint
to verify no degradation across the pipeline.

Usage:
    # Evaluate a specific checkpoint
    python scripts/evaluate.py --checkpoint_path THUDM/glm-4-9b-chat --eval humaneval

    # Evaluate all stages
    python scripts/evaluate.py --eval all

    # Compare across checkpoints
    python scripts/evaluate.py --compare
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.run_humaneval import evaluate_humaneval
from evaluation.run_swebench_lite import evaluate_swebench_lite

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

STAGE_CHECKPOINTS = {
    "base": "THUDM/glm-4-9b-chat",
    "stage1_cpt": "./checkpoints/stage1_cpt",
    "stage2_sft": "./checkpoints/stage2_sft",
    "stage3_rl": "./checkpoints/stage3_rl",
}


def run_evaluation(checkpoint_path: str, eval_type: str, output_dir: str) -> dict:
    """Run evaluation on a single checkpoint."""
    results = {"checkpoint": checkpoint_path, "timestamp": datetime.now().isoformat()}

    if eval_type in ("humaneval", "all"):
        logger.info("=== Running HumanEval on %s ===", checkpoint_path)
        try:
            he_results = evaluate_humaneval(checkpoint_path)
            results["humaneval"] = {
                "pass_rate": he_results["pass_rate"],
                "passed": he_results["passed"],
                "total": he_results["total"],
            }
            logger.info("HumanEval: %.1f%% (%d/%d)",
                        he_results["pass_rate"] * 100,
                        he_results["passed"],
                        he_results["total"])
        except Exception as e:
            logger.error("HumanEval evaluation failed: %s", e)
            results["humaneval"] = {"error": str(e)}

    if eval_type in ("agent", "all"):
        logger.info("=== Running Agent Eval on %s ===", checkpoint_path)
        try:
            agent_results = evaluate_swebench_lite(
                checkpoint_path,
                task_source="builtin",
                max_turns=20,
            )
            results["agent_eval"] = {
                "pass_rate": agent_results["pass_rate"],
                "passed": agent_results["passed"],
                "total": agent_results["total"],
            }
            logger.info("Agent Eval: %.1f%% (%d/%d)",
                        agent_results["pass_rate"] * 100,
                        agent_results["passed"],
                        agent_results["total"])
        except Exception as e:
            logger.error("Agent evaluation failed: %s", e)
            results["agent_eval"] = {"error": str(e)}

    # Save results
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    checkpoint_name = Path(checkpoint_path).name or "base"
    result_file = out_path / f"eval_{checkpoint_name}_{eval_type}.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", result_file)

    return results


def compare_stages(output_dir: str):
    """Compare evaluation results across all pipeline stages."""
    out_path = Path(output_dir)
    all_results = {}

    for stage, checkpoint in STAGE_CHECKPOINTS.items():
        result_files = list(out_path.glob(f"eval_{Path(checkpoint).name}*.json"))
        if not result_files:
            logger.info("No results found for %s, skipping", stage)
            continue
        with open(result_files[0]) as f:
            all_results[stage] = json.load(f)

    if not all_results:
        logger.warning("No evaluation results found. Run evaluations first.")
        return

    # Print comparison table
    print("\n" + "=" * 70)
    print("Pipeline Evaluation Comparison")
    print("=" * 70)
    print(f"{'Stage':<15} {'HumanEval':<20} {'Agent Eval':<20}")
    print("-" * 70)

    for stage, results in all_results.items():
        he = results.get("humaneval", {})
        ae = results.get("agent_eval", {})

        he_str = f"{he.get('pass_rate', 0) * 100:.1f}%" if "pass_rate" in he else "N/A"
        ae_str = f"{ae.get('pass_rate', 0) * 100:.1f}%" if "pass_rate" in ae else "N/A"

        print(f"{stage:<15} {he_str:<20} {ae_str:<20}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Stage 4: Evaluation")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to model checkpoint to evaluate")
    parser.add_argument("--eval", choices=["humaneval", "agent", "all"], default="all",
                        help="Which evaluation to run")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                        help="Directory to save results")
    parser.add_argument("--compare", action="store_true",
                        help="Compare results across all stages")
    parser.add_argument("--eval_all_stages", action="store_true",
                        help="Evaluate all pipeline stage checkpoints")
    args = parser.parse_args()

    if args.compare:
        compare_stages(args.output_dir)
        return

    if args.eval_all_stages:
        for stage, checkpoint in STAGE_CHECKPOINTS.items():
            ckpt_path = Path(checkpoint)
            if not (ckpt_path.exists() or "/" not in checkpoint):
                logger.warning("Checkpoint not found for %s (%s), skipping", stage, checkpoint)
                continue
            logger.info("=== Evaluating %s: %s ===", stage, checkpoint)
            run_evaluation(checkpoint, args.eval, args.output_dir)
        compare_stages(args.output_dir)
        return

    if args.checkpoint_path is None:
        args.checkpoint_path = STAGE_CHECKPOINTS["base"]
        logger.info("No checkpoint specified, using base model: %s", args.checkpoint_path)

    run_evaluation(args.checkpoint_path, args.eval, args.output_dir)


if __name__ == "__main__":
    main()
