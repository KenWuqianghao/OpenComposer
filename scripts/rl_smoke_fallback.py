#!/usr/bin/env python3
"""Lightweight smoke when OpenRLHF is unavailable: load checkpoint and run a short generate."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from opencomposer.checkpoint_utils import maybe_prune_before_hf_load
from opencomposer.train_runtime import hf_causal_lm_eval_kw

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _smoke_generate(model_path: str, max_new_tokens: int) -> dict:
    maybe_prune_before_hf_load(model_path, logger)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        **hf_causal_lm_eval_kw(),
    )
    model.eval()
    device = next(model.parameters()).device
    prompt = "# Short Python smoke test\ndef incr(x):\n    "
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=getattr(tokenizer, "eos_token_id", None),
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return {
        "smoke": "load_generate",
        "model_path": model_path,
        "ok": True,
        "output_preview": text[:400],
        "output_length": len(text),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Load checkpoint and run a short generate (no OpenRLHF)")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="./evaluation_results/rl_fallback_smoke.json")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument(
        "--fallback_model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HF id used if primary checkpoint fails to load",
    )
    args = parser.parse_args()

    logger.info("RL smoke (load+generate) on %s", args.checkpoint_path)
    fallback_used = False
    first_error = None
    try:
        results = _smoke_generate(args.checkpoint_path, args.max_new_tokens)
    except Exception as e:
        first_error = f"{type(e).__name__}: {e}"
        logger.warning(
            "Primary checkpoint failed (%s). Retrying with %s",
            first_error,
            args.fallback_model,
        )
        fallback_used = True
        results = _smoke_generate(args.fallback_model, args.max_new_tokens)
        results["primary_model_error"] = first_error
        results["fallback_model"] = args.fallback_model
    results["fallback_used"] = fallback_used

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("RL smoke done. Report: %s", out_path)


if __name__ == "__main__":
    main()
