"""Custom reward function for OpenRLHF single-turn mode (alternative to agent mode).

This can be used with --remote_rm_url for simpler single-turn reward computation.
For the full multi-turn agent RL, use agent_func.py instead.
"""

from __future__ import annotations

import json
import re
import logging

logger = logging.getLogger(__name__)


def reward_fn(queries: list[str], responses: list[str], **kwargs) -> list[float]:
    """Compute rewards for single-turn coding responses.

    This is a heuristic reward function for use when not running the full
    multi-turn agent loop. It evaluates response quality based on:
    - Tool usage patterns (uses correct tool call format)
    - Presence of code in the response
    - Reasoning structure (mentions investigation, testing)

    For the full pipeline, use the multi-turn agent_func.py with
    environment-based rewards (test execution).
    """
    rewards = []
    for query, response in zip(queries, responses):
        score = 0.0

        # Reward proper tool call format
        if "<tool_call>" in response and "</tool_call>" in response:
            score += 0.3
            # Valid JSON in tool call
            match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", response, re.DOTALL)
            if match:
                try:
                    call = json.loads(match.group(1))
                    if "name" in call and "arguments" in call:
                        score += 0.2
                except json.JSONDecodeError:
                    pass

        # Reward reasoning before action
        tool_call_pos = response.find("<tool_call>")
        if tool_call_pos > 50:
            score += 0.1

        # Reward running tests
        if "pytest" in response or "run_terminal" in response:
            score += 0.1

        # Penalize very short or very long responses
        resp_len = len(response)
        if resp_len < 20:
            score -= 0.3
        elif resp_len > 5000:
            score -= 0.1

        # Penalize refusal or uncertainty
        refusal_patterns = ["i cannot", "i'm unable", "i don't know how"]
        if any(p in response.lower() for p in refusal_patterns):
            score -= 0.3

        rewards.append(max(-1.0, min(1.0, score)))

    return rewards
