"""Attach an MTP head next to a HF causal LM (optional module)."""

from __future__ import annotations

import types
from typing import Any

import torch
import torch.nn as nn

from opencomposer.mtp.mtp_head import MTPHead


def attach_mtp_head(
    model: nn.Module,
    *,
    rms_eps: float | None = None,
) -> MTPHead:
    """Add ``model.mtp_head`` and keep original ``forward`` untouched.

    Training loops should call ``mtp_head(model_last_hidden)`` explicitly.
    """
    cfg = model.config
    hs = cfg.hidden_size
    vs = cfg.vocab_size
    eps = rms_eps if rms_eps is not None else getattr(cfg, "rms_norm_eps", 1e-6)
    head = MTPHead(hs, vs, rms_eps=eps)
    model.mtp_head = head  # type: ignore[attr-defined]
    return head


def tie_mtp_head_to_embeddings(model: nn.Module, mtp: MTPHead) -> None:
    """Optional: tie output projection to input embeddings."""
    emb = getattr(model, "get_input_embeddings", lambda: None)()
    if emb is not None and hasattr(emb, "weight"):
        mtp.lm_head.weight = emb.weight  # type: ignore[assignment]
