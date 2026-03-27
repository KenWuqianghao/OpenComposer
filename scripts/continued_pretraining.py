#!/usr/bin/env python3
"""Stage 1: Continued pretraining of GLM-4-9B on code data.

Trains the base GLM-4-9B model on a code-focused corpus using standard causal
language modeling with DeepSpeed ZeRO-2.

Usage:
    deepspeed scripts/continued_pretraining.py \
        --deepspeed configs/deepspeed_zero2.json \
        --config configs/continued_pretraining.yaml

    # Or without DeepSpeed (single GPU):
    python scripts/continued_pretraining.py \
        --config configs/continued_pretraining.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from opencomposer.checkpoint_utils import (
    PruneStaleMonolithicCallback,
    maybe_prune_before_hf_load,
    prune_stale_monolithic_under,
)
from opencomposer.data.code_dataset import create_code_pretraining_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Continued pretraining on code")
    parser.add_argument("--config", type=str, default="configs/continued_pretraining.yaml")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--max_steps", type=int, default=None, help="Override max_steps from config")
    # Allow DeepSpeed to pass its args
    parser.add_argument("--deepspeed", type=str, default=None)
    args, unknown = parser.parse_known_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    if args.deepspeed:
        train_cfg["deepspeed"] = args.deepspeed
    if args.max_steps is not None:
        train_cfg["max_steps"] = args.max_steps

    logger.info("Loading tokenizer: %s", model_cfg["name_or_path"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name_or_path"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn = model_cfg.get("attn_implementation", "sdpa")
    logger.info("Loading model: %s (attn=%s)", model_cfg["name_or_path"], attn)
    maybe_prune_before_hf_load(model_cfg["name_or_path"], logger)
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

    logger.info("Preparing code pretraining dataset")
    train_dataset = create_code_pretraining_dataset(
        tokenizer=tokenizer,
        dataset_name=data_cfg.get("dataset_name", "bigcode/starcoderdata"),
        languages=data_cfg.get("languages"),
        max_seq_length=data_cfg.get("max_seq_length", 4096),
        streaming=data_cfg.get("streaming", True),
        fallback_dataset=data_cfg.get("fallback_dataset"),
        fallback_dataset_config=data_cfg.get("fallback_dataset_config"),
        fallback_text_column=data_cfg.get("fallback_text_column", "text"),
    )

    ta = dict(
        output_dir=train_cfg["output_dir"],
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 64),
        learning_rate=train_cfg.get("learning_rate", 2e-5),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
        num_train_epochs=train_cfg.get("num_train_epochs", 1),
        max_steps=train_cfg.get("max_steps", 5000),
        save_steps=train_cfg.get("save_steps", 500),
        logging_steps=train_cfg.get("logging_steps", 10),
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 4),
        report_to="wandb" if train_cfg.get("use_wandb") else "none",
        save_total_limit=3,
        remove_unused_columns=False,
        max_grad_norm=1.0,
    )
    ds_path = train_cfg.get("deepspeed")
    if ds_path:
        ta["deepspeed"] = ds_path
    if train_cfg.get("optim"):
        ta["optim"] = train_cfg["optim"]
    training_args = TrainingArguments(**ta)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[PruneStaleMonolithicCallback()],
    )

    logger.info("Starting continued pretraining")
    trainer.train()

    logger.info("Saving final checkpoint")
    trainer.save_model(train_cfg["output_dir"])
    if prune_stale_monolithic_under(train_cfg["output_dir"]):
        logger.info(
            "Removed stale model.safetensors where sharded weights exist "
            "so from_pretrained loads ChatGLM transformer.* tensors.",
        )
    tokenizer.save_pretrained(train_cfg["output_dir"])
    logger.info("Continued pretraining complete. Model saved to %s", train_cfg["output_dir"])


if __name__ == "__main__":
    main()
