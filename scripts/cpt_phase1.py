#!/usr/bin/env python3
"""Stage 1a: CPT phase 1 (short context) for Qwen MoE."""

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
from opencomposer.train_runtime import apply_cpu_safe_training_defaults, hard_exit_clean, resolve_torch_dtype

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def load_yaml(p: str) -> dict:
    with open(p) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="CPT phase 1 (MoE / Qwen)")
    parser.add_argument("--config", type=str, default="configs/qwen3_moe_cpt_phase1.yaml")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--deepspeed", type=str, default=None)
    args, _ = parser.parse_known_args()

    cfg = load_yaml(args.config)
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    if args.deepspeed:
        train_cfg["deepspeed"] = args.deepspeed
    if args.max_steps is not None:
        train_cfg["max_steps"] = int(args.max_steps)

    apply_cpu_safe_training_defaults(model_cfg, train_cfg)

    tok_path = model_cfg["name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn = model_cfg.get("attn_implementation", "sdpa")
    maybe_prune_before_hf_load(tok_path, logger)
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

    train_dataset = create_code_pretraining_dataset(
        tokenizer=tokenizer,
        dataset_name=data_cfg.get("dataset_name", "bigcode/starcoderdata"),
        languages=data_cfg.get("languages"),
        max_seq_length=data_cfg.get("max_seq_length", 2048),
        streaming=data_cfg.get("streaming", True),
        fallback_dataset=data_cfg.get("fallback_dataset"),
        fallback_dataset_config=data_cfg.get("fallback_dataset_config"),
        fallback_text_column=data_cfg.get("fallback_text_column", "text"),
    )

    ta = dict(
        output_dir=train_cfg["output_dir"],
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 32),
        learning_rate=train_cfg.get("learning_rate", 2e-5),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
        num_train_epochs=train_cfg.get("num_train_epochs", 1),
        max_steps=train_cfg.get("max_steps", 5000),
        save_steps=train_cfg.get("save_steps", 500),
        logging_steps=train_cfg.get("logging_steps", 10),
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 0),
        report_to="wandb" if train_cfg.get("use_wandb") else "none",
        save_total_limit=3,
        remove_unused_columns=False,
        max_grad_norm=1.0,
    )
    if train_cfg.get("deepspeed"):
        ta["deepspeed"] = train_cfg["deepspeed"]
    if train_cfg.get("optim"):
        ta["optim"] = train_cfg["optim"]
    training_args = TrainingArguments(**ta)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        tokenizer=tokenizer,
        callbacks=[PruneStaleMonolithicCallback()],
    )
    trainer.train()
    trainer.save_model(train_cfg["output_dir"])
    prune_stale_monolithic_under(train_cfg["output_dir"])
    tokenizer.save_pretrained(train_cfg["output_dir"])
    logger.info("Saved to %s", train_cfg["output_dir"])
    hard_exit_clean()


if __name__ == "__main__":
    main()
