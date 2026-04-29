#!/usr/bin/env python3
"""Stage 2: SFT for tool-use (Qwen ChatML)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import yaml
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from opencomposer.checkpoint_utils import (
    PruneStaleMonolithicCallback,
    maybe_prune_before_hf_load,
    prune_stale_monolithic_under,
)
from opencomposer.data.sft_dataset import load_sft_dataset
from opencomposer.train_runtime import apply_cpu_safe_training_defaults, hard_exit_clean, resolve_torch_dtype

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _norm_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "tool":
            out.append({"role": "user", "content": content})
        else:
            out.append({"role": role, "content": content})
    return out


class SFTDataCollator:
    """Chat-template SFT with assistant-only loss when mask is available."""

    def __init__(self, tokenizer, max_seq_length: int = 8192):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []

        for example in examples:
            messages = _norm_messages(example["messages"])
            input_ids, labels = self._encode_with_assistant_mask(messages)
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_attention_mask.append(torch.ones_like(input_ids))

        max_len = min(max(len(ids) for ids in batch_input_ids), self.max_seq_length)
        padded_input_ids = []
        padded_labels = []
        padded_attention_mask = []
        pad_id = self.tokenizer.pad_token_id or 0

        for ids, lab, mask in zip(batch_input_ids, batch_labels, batch_attention_mask):
            pad_len = max_len - len(ids)
            if pad_len > 0:
                padded_input_ids.append(torch.cat([ids, torch.full((pad_len,), pad_id, dtype=ids.dtype)]))
                padded_labels.append(torch.cat([lab, torch.full((pad_len,), -100, dtype=lab.dtype)]))
                padded_attention_mask.append(torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)]))
            else:
                padded_input_ids.append(ids[:max_len])
                padded_labels.append(lab[:max_len])
                padded_attention_mask.append(mask[:max_len])

        return {
            "input_ids": torch.stack(padded_input_ids),
            "labels": torch.stack(padded_labels),
            "attention_mask": torch.stack(padded_attention_mask),
        }

    def _encode_with_assistant_mask(self, messages: list[dict[str, str]]) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            tok_out = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                add_generation_prompt=False,
                return_assistant_tokens_mask=True,
            )
            ids = tok_out["input_ids"]
            mask = tok_out.get("assistant_masks")
            if isinstance(ids[0], list):
                ids = ids[0]
            if mask is not None and len(mask) == len(ids) and any(mask):
                labels = [tid if m else -100 for tid, m in zip(ids, mask)]
            else:
                labels = self._fallback_labels(messages, ids)
        except Exception:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            enc = self.tokenizer(text, add_special_tokens=False, return_attention_mask=False)
            ids = enc["input_ids"]
            labels = self._fallback_labels(messages, ids)
        if not ids:
            ids = [self.tokenizer.eos_token_id or 0]
            labels = [-100]
        input_ids = torch.tensor(ids[: self.max_seq_length], dtype=torch.long)
        label_ids = torch.tensor(labels[: self.max_seq_length], dtype=torch.long)
        if torch.all(label_ids == -100):
            label_ids[-1] = input_ids[-1]
        return input_ids, label_ids

    def _fallback_labels(self, messages: list[dict[str, str]], ids: list[int]) -> list[int]:
        """Prefer training on the tail of the sequence (assistant completion) if mask unavailable."""
        if not ids:
            return []
        # Last 25% of tokens — usually contains the final assistant reply in our synthetic data
        cut = int(len(ids) * 0.72)
        return [-100] * cut + ids[cut:]


def main():
    parser = argparse.ArgumentParser(description="Stage 2: SFT for tool-use (Qwen ChatML)")
    parser.add_argument("--config", type=str, default="configs/qwen3_moe_sft_toolu.yaml")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed", type=str, default=None)
    args, _ = parser.parse_known_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    if args.deepspeed:
        train_cfg["deepspeed"] = args.deepspeed

    apply_cpu_safe_training_defaults(model_cfg, train_cfg)

    tok_path = model_cfg["name_or_path"]
    logger.info("Loading tokenizer from: %s", tok_path)
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model from: %s", tok_path)
    maybe_prune_before_hf_load(tok_path, logger)
    attn = model_cfg.get("attn_implementation", "sdpa")
    model_config = AutoConfig.from_pretrained(tok_path, trust_remote_code=True)
    if not hasattr(model_config, "max_length") and hasattr(model_config, "seq_length"):
        model_config.max_length = model_config.seq_length
    model = AutoModelForCausalLM.from_pretrained(
        tok_path,
        config=model_config,
        torch_dtype=resolve_torch_dtype(model_cfg),
        attn_implementation=attn,
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()

    train_dataset = load_sft_dataset(data_cfg["dataset_path"], split="train")
    eval_dataset = load_sft_dataset(data_cfg["dataset_path"], split="test")
    logger.info("Train: %d examples, Eval: %d examples", len(train_dataset), len(eval_dataset))

    data_collator = SFTDataCollator(tokenizer=tokenizer, max_seq_length=data_cfg.get("max_seq_length", 8192))

    ta_kwargs = dict(
        output_dir=train_cfg["output_dir"],
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 32),
        learning_rate=train_cfg.get("learning_rate", 1e-5),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
        num_train_epochs=train_cfg.get("num_train_epochs", 2),
        save_steps=train_cfg.get("save_steps", 200),
        logging_steps=train_cfg.get("logging_steps", 5),
        eval_strategy="steps",
        eval_steps=train_cfg.get("save_steps", 200),
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        report_to="wandb" if train_cfg.get("use_wandb") else "none",
        save_total_limit=3,
        remove_unused_columns=False,
        max_grad_norm=1.0,
        load_best_model_at_end=not train_cfg.get("no_eval", False),
        metric_for_best_model="eval_loss",
    )
    if train_cfg.get("no_eval"):
        ta_kwargs["eval_strategy"] = "no"
        ta_kwargs.pop("eval_steps", None)
        ta_kwargs["load_best_model_at_end"] = False
    if train_cfg.get("max_steps") is not None:
        ta_kwargs["max_steps"] = int(train_cfg["max_steps"])
    if train_cfg.get("dataloader_num_workers") is not None:
        ta_kwargs["dataloader_num_workers"] = int(train_cfg["dataloader_num_workers"])
    if train_cfg.get("optim"):
        ta_kwargs["optim"] = train_cfg["optim"]
    ds_path = train_cfg.get("deepspeed")
    if ds_path:
        ta_kwargs["deepspeed"] = ds_path
    else:
        ta_kwargs.pop("deepspeed", None)

    training_args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[PruneStaleMonolithicCallback()],
    )

    logger.info("Starting SFT training")
    trainer.train()

    trainer.save_model(train_cfg["output_dir"])
    prune_stale_monolithic_under(train_cfg["output_dir"])
    tokenizer.save_pretrained(train_cfg["output_dir"])
    logger.info("SFT complete. Model saved to %s", train_cfg["output_dir"])
    hard_exit_clean()


if __name__ == "__main__":
    main()
