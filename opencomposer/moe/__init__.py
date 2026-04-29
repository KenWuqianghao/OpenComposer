"""Mixture-of-Experts utilities: expert traces and router replay (Composer 2-style)."""

from opencomposer.moe.expert_trace import ExpertTrace, ExpertTraceBuilder
from opencomposer.moe.router_replay import RouterReplayController, router_replay_context
from opencomposer.moe.qwen3_moe_patch import apply_moe_router_hooks, remove_moe_router_hooks
from opencomposer.moe.aux_loss import set_moe_aux_loss_enabled

__all__ = [
    "ExpertTrace",
    "ExpertTraceBuilder",
    "RouterReplayController",
    "router_replay_context",
    "apply_moe_router_hooks",
    "remove_moe_router_hooks",
    "set_moe_aux_loss_enabled",
]
