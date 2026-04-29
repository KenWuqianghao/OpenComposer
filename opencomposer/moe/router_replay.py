"""Thread-local controller for capture / replay of MoE router decisions."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Generator, Literal

import torch

from opencomposer.moe.expert_trace import ExpertTrace, ExpertTraceBuilder

Mode = Literal["none", "capture", "replay"]

_ctx_mode: ContextVar[str] = ContextVar("opencomposer_moe_router_mode", default="none")
_ctx_trace_builder: ContextVar[ExpertTraceBuilder | None] = ContextVar(
    "opencomposer_moe_trace_builder", default=None
)
_ctx_replay_trace: ContextVar[ExpertTrace | None] = ContextVar(
    "opencomposer_moe_replay_trace", default=None
)
_ctx_tau: ContextVar[float] = ContextVar("opencomposer_moe_replay_tau", default=0.2)


class RouterReplayController:
    """Access from patched MoE forwards."""

    @staticmethod
    def mode() -> Mode:
        m = _ctx_mode.get()
        if m in ("none", "capture", "replay"):
            return m  # type: ignore[return-value]
        return "none"

    @staticmethod
    def tau() -> float:
        return float(_ctx_tau.get())

    @staticmethod
    def builder() -> ExpertTraceBuilder | None:
        return _ctx_trace_builder.get()

    @staticmethod
    def replay_trace() -> ExpertTrace | None:
        return _ctx_replay_trace.get()

    @staticmethod
    def merge_with_plausibility(
        router_probs: torch.Tensor,
        live_sel: torch.Tensor,
        replay_sel: torch.Tensor,
        *,
        tau: float,
        norm_topk: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build final (selected_experts, routing_weights) using replay + plausibility filter."""
        top1 = router_probs.max(dim=-1, keepdim=True).values
        thresh = tau * top1
        out_sel = replay_sel.clone()
        for i in range(out_sel.shape[0]):
            for j in range(out_sel.shape[1]):
                e = int(replay_sel[i, j].item())
                if router_probs[i, e] < thresh[i, 0]:
                    out_sel[i, j] = live_sel[i, j]
        rw = router_probs.gather(1, out_sel.long())
        if norm_topk:
            rw = rw / rw.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return out_sel, rw


@contextmanager
def router_replay_context(
    mode: Mode,
    *,
    trace_builder: ExpertTraceBuilder | None = None,
    replay_trace: ExpertTrace | None = None,
    tau: float = 0.2,
) -> Generator[None, None, None]:
    """Set router replay mode for patched MoE blocks in this context."""
    tok_mode = _ctx_mode.set(mode)
    tok_builder = _ctx_trace_builder.set(trace_builder)
    tok_replay = _ctx_replay_trace.set(replay_trace)
    tok_tau = _ctx_tau.set(tau)
    try:
        yield
    finally:
        _ctx_tau.reset(tok_tau)
        _ctx_replay_trace.reset(tok_replay)
        _ctx_trace_builder.reset(tok_builder)
        _ctx_mode.reset(tok_mode)
