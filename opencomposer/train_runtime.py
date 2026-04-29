"""Shared training/eval device and dtype defaults (CPU smoke vs GPU)."""

from __future__ import annotations

import os
from typing import Any

import torch


def use_cpu_training() -> bool:
    """True when we should force CPU-safe dtype/optim (no bf16, no bitsandbytes)."""
    if os.environ.get("OPENCOMPOSER_FORCE_CPU", "").lower() in ("1", "true", "yes"):
        return True
    return not torch.cuda.is_available()


def apply_cpu_safe_training_defaults(model_cfg: dict[str, Any], train_cfg: dict[str, Any]) -> None:
    if not use_cpu_training():
        return
    model_cfg["torch_dtype"] = "float32"
    model_cfg["attn_implementation"] = "eager"
    train_cfg["bf16"] = False
    train_cfg["deepspeed"] = None
    if train_cfg.get("optim") == "adamw_bnb_8bit":
        train_cfg["optim"] = "adamw_torch"


def hf_causal_lm_eval_kw() -> dict[str, Any]:
    """Keyword args for ``AutoModelForCausalLM.from_pretrained`` in evaluation harnesses."""
    if torch.cuda.is_available():
        return {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    return {"torch_dtype": torch.float32, "device_map": None}


def resolve_torch_dtype(model_cfg: dict[str, Any]):
    name = model_cfg.get("torch_dtype", "bfloat16")
    if isinstance(name, str):
        return getattr(torch, name)
    return name


def hard_exit_clean() -> None:
    """Bypass flaky native shutdown (streaming datasets + PyTorch) when requested."""
    if os.environ.get("OPENCOMPOSER_EXIT_HARD", "").lower() in ("1", "true", "yes"):
        os._exit(0)
