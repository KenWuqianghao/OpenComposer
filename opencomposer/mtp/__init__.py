"""MTP self-distillation (Composer 2 mini)."""

from opencomposer.mtp.mtp_head import MTPHead
from opencomposer.mtp.attach_mtp import attach_mtp_head, tie_mtp_head_to_embeddings
from opencomposer.mtp.self_distill import kl_logits_loss, mtp_self_distill_step

__all__ = [
    "MTPHead",
    "attach_mtp_head",
    "tie_mtp_head_to_embeddings",
    "kl_logits_loss",
    "mtp_self_distill_step",
]
