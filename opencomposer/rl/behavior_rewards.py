"""Heuristic behavioral auxiliary rewards (Composer 2 §4.2 mini)."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class BehaviorRewardConfig:
    style_weight: float = 0.05
    comms_weight: float = 0.05
    tool_quality_weight: float = 0.05


def _style_penalty(text: str) -> float:
    """Penalize comment-as-chain-of-thought and noisy debug prints."""
    penalty = 0.0
    if re.search(r"```[\s\S]{800,}", text):
        penalty += 0.1
    if len(re.findall(r"\bprint\s*\(", text)) > 5:
        penalty += 0.1
    if re.search(r"#\s*(TODO|FIXME).*\n.*\n.*#", text, re.I):
        penalty += 0.05
    return penalty


def _comms_penalty(text: str) -> float:
    emoji = len(re.findall(r"[\U00010000-\U0010ffff]", text))  # wide unicode
    if emoji > 10:
        return min(0.2, 0.01 * (emoji - 10))
    if text.count("**") > 8:
        return 0.05
    # Prefix-stream / stutter: many lines starting same short prefix
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 6:
        heads = [ln[:20] for ln in lines[:30]]
        if len(set(heads)) <= 3:
            return 0.1
    return 0.0


def _tool_quality_penalty(text: str) -> float:
    p = 0.0
    if text.count("<tool_call>") > text.count("</tool_call>"):
        p += 0.1
    broken_json = len(re.findall(r"<tool_call>\s*\{[^}]*$", text, re.M))
    p += 0.05 * min(broken_json, 3)
    calls = re.findall(r"<tool_call>\s*(\{[^<]*\})\s*</tool_call>", text, re.DOTALL)
    seen = set()
    for c in calls:
        if c in seen:
            p += 0.03
        seen.add(c)
    todos = len(re.findall(r"\bTODO\b", text, re.I))
    dones = len(re.findall(r"\bdone\b|\bcompleted\b", text, re.I))
    if todos > 2 and dones == 0 and len(text) > 500:
        p += 0.05
    return min(0.3, p)


def behavior_aux_score(
    trajectory_text: str,
    cfg: BehaviorRewardConfig | None = None,
    *,
    enabled: tuple[str, ...] = ("style", "comms", "tool_quality"),
) -> float:
    """Return additive reward delta in roughly [-0.2, 0.15]."""
    cfg = cfg or BehaviorRewardConfig()
    total = 0.0
    if "style" in enabled:
        total -= cfg.style_weight * _style_penalty(trajectory_text)
    if "comms" in enabled:
        total -= cfg.comms_weight * _comms_penalty(trajectory_text)
    if "tool_quality" in enabled:
        total -= cfg.tool_quality_weight * _tool_quality_penalty(trajectory_text)
    return float(total)
