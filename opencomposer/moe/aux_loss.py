"""Helpers to toggle Qwen MoE load-balancing auxiliary loss (if present)."""

from __future__ import annotations

import torch.nn as nn


def set_moe_aux_loss_enabled(model: nn.Module, enabled: bool) -> None:
    """Best-effort: set router_aux_loss_coef or similar on config when available."""
    config = getattr(model, "config", None)
    if config is None:
        return
    # Qwen2/3 MoE: router_aux_loss_coef on config
    if hasattr(config, "router_aux_loss_coef"):
        try:
            if not enabled:
                setattr(model.config, "_opencomposer_saved_aux_coef", config.router_aux_loss_coef)
                config.router_aux_loss_coef = 0.0
            else:
                saved = getattr(model.config, "_opencomposer_saved_aux_coef", None)
                if saved is not None:
                    config.router_aux_loss_coef = saved
        except Exception:
            pass
