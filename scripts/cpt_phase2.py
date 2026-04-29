#!/usr/bin/env python3
"""Stage 1b: CPT with long-context RoPE / YARN overrides."""

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
    parser = argparse.ArgumentParser(description="CPT phase 2 long context")
    parser.add_argument("--config", type=str, default="configs/qwen3_moe_cpt_phase2.yaml")
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

    base = model_cfg["name_or_path"]
    maybe_prune_before_hf_load(base, logger)
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_config = AutoConfig.from_pretrained(base, trust_remote_code=True)
    if model_cfg.get("max_position_embeddings"):
        setattr(model_config, "max_position_embeddings", int(model_cfg["max_position_embeddings"]))
    if model_cfg.get("rope_scaling") is not None:
        try:
            setattr(model_config, "rope_scaling", model_cfg["rope_scaling"])
        except Exception as e:
            logger.warning("Could not set rope_scaling on config: %s", e)

    attn = model_cfg.get("attn_implementation", "sdpa")
    model = AutoModelForCausalLM.from_pretrained(
        base,
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
        max_seq_length=data_cfg.get("max_seq_length", 4096),
        streaming=data_cfg.get("streaming", True),
        fallback_dataset=data_cfg.get("fallback_dataset"),
        fallback_dataset_config=data_cfg.get("fallback_dataset_config"),
        fallback_text_column=data_cfg.get("fallback_text_column", "text"),
    )

    ta = dict(
        output_dir=train_cfg["output_dir"],
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 48),
        learning_rate=train_cfg.get("learning_rate", 1e-5),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.02),
        num_train_epochs=train_cfg.get("num_train_epochs", 1),
        max_steps=train_cfg.get("max_steps", 1000),
        save_steps=train_cfg.get("save_steps", 200),
        logging_steps=train_cfg.get("logging_steps", 5),
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 0),
        report_to="wandb" if train_cfg.get("use_wandb") else "none",
        save_total_limit=2,
        remove_unused_columns=False,
        max_grad_norm=1.0,
    )
    if train_cfg.get("deepspeed"):
        ta["deepspeed"] = train_cfg["deepspeed"]
    if train_cfg.get("optim"):
        ta["optim"] = train_cfg["optim"]

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**ta),
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
