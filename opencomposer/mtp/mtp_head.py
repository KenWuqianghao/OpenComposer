"""Multi-token prediction head (Composer 2 §3.1 mini variant)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.variance_epsilon)
        return (self.weight * x).to(dtype)


class MTPHead(nn.Module):
    """Predict logits at t+1 from hidden state at t (self-distill target: main LM head)."""

    def __init__(self, hidden_size: int, vocab_size: int, rms_eps: float = 1e-6):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=rms_eps)
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.act = nn.SiLU()
        self.out_norm = RMSNorm(hidden_size, eps=rms_eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self._init_weights(hidden_size, vocab_size)

    def _init_weights(self, hidden_size: int, vocab_size: int) -> None:
        std = 1.0 / math.sqrt(hidden_size)
        nn.init.trunc_normal_(self.fc.weight, std=std)
        nn.init.trunc_normal_(self.lm_head.weight, std=std)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.norm(hidden_states)
        x = self.fc(x)
        x = self.act(x)
        x = self.out_norm(x)
        return self.lm_head(x)
