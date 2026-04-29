"""Composer 2 nonlinear length penalty C_{k,q}(x) (§4.2)."""

from __future__ import annotations

from dataclasses import dataclass


def composer_length_penalty(x: float, k: float, q: float) -> float:
    r"""C_{k,q}(x) = ((1 + k x)^{1-q} - 1) / (k (1 - q)).

    For q != 1. When q → 1, use logarithmic limit (not implemented here—pick q != 1).
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if abs(q - 1.0) < 1e-9:
        raise ValueError("q must differ from 1 for this closed form")
    return ((1.0 + k * x) ** (1.0 - q) - 1.0) / (k * (1.0 - q))


@dataclass
class LengthPenaltyWeights:
    thinking_tokens: float = 0.1
    tool_call_tokens: float = 0.2
    tool_output_tokens: float = 0.15
    final_message_tokens: float = 0.1
    num_tool_calls: float = 0.5
    num_turns: float = 0.3


def compute_weighted_effort(stats: dict) -> float:
    """Map episode stats to scalar x for C_{k,q}."""
    w = LengthPenaltyWeights()
    return (
        w.thinking_tokens * float(stats.get("thinking_tokens", 0))
        + w.tool_call_tokens * float(stats.get("tool_call_tokens", 0))
        + w.tool_output_tokens * float(stats.get("tool_output_tokens", 0))
        + w.final_message_tokens * float(stats.get("final_message_tokens", 0))
        + w.num_tool_calls * float(stats.get("num_tool_calls", 0))
        + w.num_turns * float(stats.get("num_turns", 0))
    )


def episode_length_penalty_reward_delta(
    stats: dict,
    *,
    k: float,
    q: float,
    scale: float = 0.02,
) -> float:
    """Return *negative* penalty term to add to reward (scaled)."""
    x = compute_weighted_effort(stats)
    pen = composer_length_penalty(x, k, q)
    return -float(scale) * float(pen)
