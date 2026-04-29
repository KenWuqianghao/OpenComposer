"""Self-distillation loss: MTP logits match frozen main LM logits."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_logits_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL(teacher || student) with temperature (mean over batch and length)."""
    t = temperature
    p = F.log_softmax(student_logits / t, dim=-1)
    q = F.softmax(teacher_logits / t, dim=-1)
    kl = F.kl_div(p, q, reduction="none").sum(-1)
    return (kl * (t * t)).mean()


def mtp_self_distill_step(
    model: torch.nn.Module,
    mtp_head: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """MTP(hidden[:-1]) matches LM logits[:, 1:] from frozen backbone."""
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        logits_main: torch.Tensor = out.logits[:, 1:].detach()
        hidden = out.hidden_states[-1][:, :-1].detach()
    logits_mtp = mtp_head(hidden)
    return kl_logits_loss(logits_mtp, logits_main, temperature=temperature)
