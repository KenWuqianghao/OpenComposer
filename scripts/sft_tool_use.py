#!/usr/bin/env python3
"""Stage 2: Supervised fine-tuning for tool-use format.

Teaches the model to emit structured tool calls and reason over tool results
in multi-turn conversations.

Usage:
    deepspeed scripts/sft_tool_use.py \
        --deepspeed configs/deepspeed_zero2.json \
        --config configs/sft.yaml

    # Or without DeepSpeed:
    python scripts/sft_tool_use.py --config configs/sft.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import yaml
import torch
from datasets import Dataset
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class SFTDataCollator:
    """Collator that tokenizes chat messages and masks non-assistant tokens in the loss."""

    def __init__(self, tokenizer, max_seq_length: int = 8192):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        # Marker tokens for identifying assistant turns
        self._assistant_marker = "<|assistant|>"
        self._tool_call_start = "<tool_call>"

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []

        for example in examples:
            messages = example["messages"]
            text = self._format_messages(messages)

            encoded = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
                padding=False,
            )
            input_ids = encoded["input_ids"].squeeze(0)

            # Build labels: mask everything except assistant responses
            labels = self._build_labels(text, input_ids)

            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_attention_mask.append(torch.ones_like(input_ids))

        # Pad to longest in batch
        max_len = min(max(len(ids) for ids in batch_input_ids), self.max_seq_length)
        padded_input_ids = []
        padded_labels = []
        padded_attention_mask = []

        for ids, lab, mask in zip(batch_input_ids, batch_labels, batch_attention_mask):
            pad_len = max_len - len(ids)
            if pad_len > 0:
                pad_id = self.tokenizer.pad_token_id or 0
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

    def _format_messages(self, messages: list[dict]) -> str:
        """Format messages into a flat string with role markers."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"<|system|>\n{content}\n")
            elif role == "user":
                parts.append(f"<|user|>\n{content}\n")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}\n")
            elif role == "tool":
                parts.append(f"<|observation|>\n{content}\n")
        return "".join(parts)

    def _build_labels(self, text: str, input_ids: torch.Tensor) -> torch.Tensor:
        """Build labels tensor, masking non-assistant turns with -100."""
        labels = input_ids.clone()

        # Find assistant response regions in the text
        assistant_marker = "<|assistant|>"
        non_assistant_markers = ["<|system|>", "<|user|>", "<|observation|>"]

        in_assistant = False
        current_pos = 0
        char_mask = [False] * len(text)

        while current_pos < len(text):
            next_assistant = text.find(assistant_marker, current_pos)
            next_others = []
            for marker in non_assistant_markers:
                idx = text.find(marker, current_pos)
                if idx >= 0:
                    next_others.append(idx)

            if next_assistant >= 0 and (not next_others or next_assistant <= min(next_others)):
                # Mark from assistant marker content start to next non-assistant marker
                content_start = next_assistant + len(assistant_marker)
                end_pos = len(text)
                for marker in non_assistant_markers:
                    idx = text.find(marker, content_start)
                    if idx >= 0:
                        end_pos = min(end_pos, idx)

                for i in range(content_start, end_pos):
                    char_mask[i] = True
                current_pos = end_pos
            elif next_others:
                current_pos = min(next_others) + 1
            else:
                break

        # Map character mask to token mask (approximate: mask tokens where
        # majority of their characters are not in assistant regions)
        token_strs = []
        for i in range(len(input_ids)):
            token_strs.append(self.tokenizer.decode([input_ids[i].item()]))

        char_idx = 0
        for tok_idx, tok_str in enumerate(token_strs):
            tok_len = len(tok_str)
            if tok_len == 0:
                labels[tok_idx] = -100
                continue

            end_char = min(char_idx + tok_len, len(char_mask))
            if char_idx < len(char_mask):
                in_region = sum(char_mask[char_idx:end_char])
                if in_region < tok_len / 2:
                    labels[tok_idx] = -100
            else:
                labels[tok_idx] = -100
            char_idx = end_char

        return labels


def main():
    parser = argparse.ArgumentParser(description="Stage 2: SFT for tool-use")
    parser.add_argument("--config", type=str, default="configs/sft.yaml")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed", type=str, default=None)
    args, unknown = parser.parse_known_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    if args.deepspeed:
        train_cfg["deepspeed"] = args.deepspeed

    logger.info("Loading tokenizer from: %s", model_cfg["name_or_path"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name_or_path"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model from: %s", model_cfg["name_or_path"])
    maybe_prune_before_hf_load(model_cfg["name_or_path"], logger)
    attn = model_cfg.get("attn_implementation", "sdpa")
    model_config = AutoConfig.from_pretrained(
        model_cfg["name_or_path"], trust_remote_code=True
    )
    if not hasattr(model_config, "max_length") and hasattr(model_config, "seq_length"):
        model_config.max_length = model_config.seq_length
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name_or_path"],
        config=model_config,
        torch_dtype=getattr(torch, model_cfg.get("torch_dtype", "bfloat16")),
        attn_implementation=attn,
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()

    logger.info("Loading SFT dataset from: %s", data_cfg["dataset_path"])
    train_dataset = load_sft_dataset(data_cfg["dataset_path"], split="train")
    eval_dataset = load_sft_dataset(data_cfg["dataset_path"], split="test")
    logger.info("Train: %d examples, Eval: %d examples", len(train_dataset), len(eval_dataset))

    data_collator = SFTDataCollator(
        tokenizer=tokenizer,
        max_seq_length=data_cfg.get("max_seq_length", 8192),
    )

    ta_kwargs = dict(
        output_dir=train_cfg["output_dir"],
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 16),
        learning_rate=train_cfg.get("learning_rate", 1e-5),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        save_steps=train_cfg.get("save_steps", 200),
        logging_steps=train_cfg.get("logging_steps", 5),
        eval_strategy="steps",
        eval_steps=train_cfg.get("save_steps", 200),
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        deepspeed=train_cfg.get("deepspeed"),
        report_to="wandb" if train_cfg.get("use_wandb") else "none",
        save_total_limit=3,
        remove_unused_columns=False,
        max_grad_norm=1.0,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
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

    logger.info("Saving final model")
    trainer.save_model(train_cfg["output_dir"])
    if prune_stale_monolithic_under(train_cfg["output_dir"]):
        logger.info(
            "Removed stale model.safetensors where sharded weights exist "
            "so from_pretrained loads ChatGLM transformer.* tensors.",
        )
    tokenizer.save_pretrained(train_cfg["output_dir"])
    logger.info("SFT complete. Model saved to %s", train_cfg["output_dir"])


if __name__ == "__main__":
    main()
