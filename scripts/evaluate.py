#!/usr/bin/env python3
"""Stage 4: Unified evaluation (Qwen MoE pipeline) — HumanEval only."""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.run_humaneval import evaluate_humaneval

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_BASE = "Qwen/Qwen3-Coder-30B-A3B"

STAGE_CHECKPOINTS = {
    "base": DEFAULT_BASE,
    "stage1_cpt": "./checkpoints/qwen3_moe_cpt_phase1",
    "stage2_sft": "./checkpoints/stage2_sft",
    "stage3_rl": "./checkpoints/stage3_rl",
}


def _checkpoint_available(checkpoint: str) -> bool:
    ckpt_path = Path(checkpoint)
    if ckpt_path.is_dir():
        return True
    return "/" in checkpoint and not checkpoint.startswith((".", "/"))


def run_evaluation(checkpoint_path: str, eval_type: str, output_dir: str) -> dict:
    results = {"checkpoint": checkpoint_path, "timestamp": datetime.now().isoformat()}

    if eval_type != "humaneval":
        raise ValueError(f"Unsupported eval_type {eval_type!r}; only 'humaneval' is supported.")

    logger.info("=== Running HumanEval on %s ===", checkpoint_path)
    try:
        he_results = evaluate_humaneval(checkpoint_path)
        results["humaneval"] = {
            "pass_rate": he_results["pass_rate"],
            "passed": he_results["passed"],
            "total": he_results["total"],
        }
        logger.info(
            "HumanEval: %.1f%% (%d/%d)",
            he_results["pass_rate"] * 100,
            he_results["passed"],
            he_results["total"],
        )
    except Exception as e:
        logger.error("HumanEval evaluation failed: %s", e)
        results["humaneval"] = {"error": str(e)}

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    checkpoint_name = Path(checkpoint_path).name or "base"
    result_file = out_path / f"eval_{checkpoint_name}_{eval_type}.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", result_file)

    return results


def _pass_rates(results: dict) -> dict[str, float | str]:
    """Extract pass_rate per suite; use 'error' key if a suite failed."""
    out: dict[str, float | str] = {}
    key = "humaneval"
    block = results.get(key, {})
    if "error" in block:
        out[key] = f"error: {block['error']}"
    elif "pass_rate" in block:
        out[key] = float(block["pass_rate"])
    else:
        out[key] = "N/A"
    return out


def head_to_head_evaluate(
    checkpoint_a: str,
    checkpoint_b: str,
    suite: str,
    output_dir: str,
    *,
    label_a: str = "A",
    label_b: str = "B",
) -> dict:
    """Run HumanEval on two checkpoints and save a comparison JSON + print a table."""
    if suite != "humaneval":
        logger.warning("Only HumanEval is supported; ignoring suite=%r and using humaneval.", suite)
        suite = "humaneval"
    logger.info("Head-to-head: %s (%s) vs %s (%s)", label_a, checkpoint_a, label_b, checkpoint_b)
    ra = run_evaluation(checkpoint_a, suite, output_dir)
    rb = run_evaluation(checkpoint_b, suite, output_dir)
    cmp_path = Path(output_dir) / f"head_to_head_{label_a}_vs_{label_b}_{suite}.json"
    cmp_path.parent.mkdir(parents=True, exist_ok=True)

    pa, pb = _pass_rates(ra), _pass_rates(rb)
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "suite": suite,
        label_a: {"checkpoint": checkpoint_a, "pass_rates": pa, "raw": ra},
        label_b: {"checkpoint": checkpoint_b, "pass_rates": pb, "raw": rb},
    }
    with open(cmp_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    logger.info("Comparison saved to %s", cmp_path)

    print("\n" + "=" * 72)
    print(f"Benchmark comparison ({suite})")
    print("=" * 72)
    print(f"{'Suite':<22} {label_a:<24} {label_b:<24}")
    print("-" * 72)
    key = "humaneval"
    va, vb = pa[key], pb[key]
    if isinstance(va, float) and isinstance(vb, float):
        delta = vb - va
        sign = "+" if delta >= 0 else ""
        extra = f" (Δ {sign}{delta * 100:.1f}%)"
    else:
        extra = ""
    print(f"{'HumanEval':<22} {str(va):<24} {str(vb):<24}{extra}")
    print("=" * 72)
    return comparison


def compare_stages(output_dir: str):
    out_path = Path(output_dir)
    all_results: dict = {}

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

    print("\n" + "=" * 50)
    print("Pipeline Evaluation Comparison")
    print("=" * 50)
    print(f"{'Stage':<15} {'HumanEval':<12}")
    print("-" * 50)

    for stage, results in all_results.items():
        he = results.get("humaneval", {})

        def fmt(d):
            return f"{d.get('pass_rate', 0) * 100:.1f}%" if "pass_rate" in d else "N/A"

        print(f"{stage:<15} {fmt(he):<12}")

    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Stage 4: Evaluation (HumanEval)")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument(
        "--suite",
        choices=["humaneval"],
        default="humaneval",
        help="Evaluation suite (HumanEval only)",
    )
    parser.add_argument("--output_dir", type=str, default="./evaluation_results")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--eval_all_stages", action="store_true")
    parser.add_argument(
        "--head_to_head",
        nargs=2,
        metavar=("CKPT_A", "CKPT_B"),
        help="Run HumanEval on two checkpoints and print/save a pass-rate comparison",
    )
    parser.add_argument(
        "--head_labels",
        nargs=2,
        metavar=("LABEL_A", "LABEL_B"),
        default=("sft_baseline", "after_rl"),
        help="Labels for head-to-head table (default: sft_baseline after_rl)",
    )
    args = parser.parse_args()

    if args.head_to_head:
        head_to_head_evaluate(
            args.head_to_head[0],
            args.head_to_head[1],
            args.suite,
            args.output_dir,
            label_a=args.head_labels[0],
            label_b=args.head_labels[1],
        )
        return

    if args.compare:
        compare_stages(args.output_dir)
        return

    if args.eval_all_stages:
        for stage, checkpoint in STAGE_CHECKPOINTS.items():
            if not _checkpoint_available(checkpoint):
                logger.warning("Checkpoint not found for %s (%s), skipping", stage, checkpoint)
                continue
            run_evaluation(checkpoint, args.suite, args.output_dir)
        compare_stages(args.output_dir)
        return

    if args.checkpoint_path is None:
        args.checkpoint_path = STAGE_CHECKPOINTS["base"]
        logger.info("No checkpoint specified, using base: %s", args.checkpoint_path)

    run_evaluation(args.checkpoint_path, args.suite, args.output_dir)


if __name__ == "__main__":
    main()
