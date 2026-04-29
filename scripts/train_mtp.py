#!/usr/bin/env python3
"""Train MTP head via self-distillation (Composer 2 §3.1 mini)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def load_cfg(p: str) -> dict:
    with open(p) as f:
        return yaml.safe_load(f)


def main():
    from opencomposer.checkpoint_utils import maybe_prune_before_hf_load
    from opencomposer.data.code_dataset import create_code_pretraining_dataset
    from opencomposer.mtp.attach_mtp import attach_mtp_head
    from opencomposer.mtp.self_distill import mtp_self_distill_step
    from opencomposer.train_runtime import apply_cpu_safe_training_defaults, hard_exit_clean, resolve_torch_dtype

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3_moe_mtp.yaml")
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    mc, dc, tc = cfg["model"], cfg["data"], cfg["training"]

    apply_cpu_safe_training_defaults(mc, {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = mc["name_or_path"]
    maybe_prune_before_hf_load(path, logger)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=resolve_torch_dtype(mc),
        attn_implementation=mc.get("attn_implementation", "sdpa"),
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    mtp = attach_mtp_head(model)
    mdtype = next(model.parameters()).dtype
    mtp.to(device=device, dtype=mdtype)
    for p in mtp.parameters():
        p.requires_grad = True

    ds = create_code_pretraining_dataset(
        tokenizer=tokenizer,
        dataset_name=dc.get("dataset_name", "bigcode/starcoderdata"),
        languages=dc.get("languages"),
        max_seq_length=dc.get("max_seq_length", 2048),
        streaming=dc.get("streaming", True),
        fallback_dataset=dc.get("fallback_dataset"),
        fallback_dataset_config=dc.get("fallback_dataset_config"),
        fallback_text_column=dc.get("fallback_text_column", "text"),
    )

    opt = AdamW(mtp.parameters(), lr=float(tc.get("learning_rate", 5e-5)))
    max_steps = int(tc.get("max_steps", 500))
    warm = max(1, int(0.05 * max_steps))
    sched = get_cosine_schedule_with_warmup(opt, warm, max_steps)
    temp = float(tc.get("temperature", 1.0))
    micro = int(tc.get("gradient_accumulation_steps", 8))

    it = iter(ds)
    step = 0
    accum = 0
    loss_accum = 0.0
    while step < max_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(ds)
            batch = next(it)
        ids = torch.tensor([batch["input_ids"]], dtype=torch.long, device=device)
        mask = torch.tensor([batch["attention_mask"]], dtype=torch.long, device=device)
        loss = mtp_self_distill_step(model, mtp, ids, mask, temperature=temp) / micro
        loss.backward()
        loss_accum += loss.item()
        accum += 1
        if accum >= micro:
            torch.nn.utils.clip_grad_norm_(mtp.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            accum = 0
            if step % int(tc.get("logging_steps", 10)) == 0:
                logger.info("step %d mtp_kl=%.4f", step, loss_accum / micro)
            loss_accum = 0.0
            if step % int(tc.get("save_steps", 100)) == 0:
                out = Path(tc["output_dir"])
                out.mkdir(parents=True, exist_ok=True)
                torch.save(mtp.state_dict(), out / f"mtp_step_{step}.pt")
                logger.info("saved %s", out / f"mtp_step_{step}.pt")

    out = Path(tc["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    torch.save(mtp.state_dict(), out / "mtp_final.pt")
    logger.info("MTP training done; weights at %s/mtp_final.pt", out)
    hard_exit_clean()


if __name__ == "__main__":
    main()
