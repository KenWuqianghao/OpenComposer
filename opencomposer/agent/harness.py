"""Agent harness: orchestrates the model's interaction with tools."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from transformers import PreTrainedTokenizerBase

from opencomposer.agent.prompts import (
    format_system_message,
    format_tool_result,
    messages_to_chatml,
    parse_tool_call,
)

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    task_description: str
    messages: list[dict[str, str]] = field(default_factory=list)
    turn: int = 0
    done: bool = False
    total_tokens: int = 0
    num_summarizations: int = 0
    full_transcript: str = ""

    def add_message_plain(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def add_message(self, role: str, content: str) -> None:
        """Append a message; ``observation`` is folded into ``user`` for ChatML."""
        if role == "observation":
            self.messages.append({"role": "user", "content": content})
        else:
            self.messages.append({"role": role, "content": content})
        self.full_transcript += content


class AgentHarness:
    def __init__(
        self,
        workspace_dir: str,
        max_turns: int = 30,
        command_timeout: int = 30,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ):
        from opencomposer.agent.tools import ToolExecutor

        self.tool_executor = ToolExecutor(workspace_dir, timeout=command_timeout)
        self.max_turns = max_turns
        self.tokenizer = tokenizer
        self.state: AgentState | None = None

    def reset(self, task_description: str) -> AgentState:
        self.state = AgentState(task_description=task_description)
        self.state.add_message("system", format_system_message())
        self.state.add_message("user", task_description)
        return self.state

    def step(self, model_output: str) -> tuple[str, bool]:
        if self.state is None:
            raise RuntimeError("Call reset() before step()")

        self.state.turn += 1
        self.state.add_message("assistant", model_output)

        completion_signals = [
            "task is complete",
            "the issue has been fixed",
            "all tests pass",
            "i have completed",
            "the fix is in place",
        ]
        lower_output = model_output.lower()
        if any(sig in lower_output for sig in completion_signals) and self.state.turn > 1:
            self.state.done = True
            return "", True

        if self.state.turn >= self.max_turns:
            self.state.done = True
            return "Maximum turns reached. Episode ending.", True

        tool_call = parse_tool_call(model_output)
        if tool_call is None:
            feedback = (
                "No valid tool call detected. Please use the <tool_call> format to call a tool, "
                "or state that the task is complete."
            )
            self.state.add_message("observation", feedback)
            return feedback, False

        tool_name, arguments = tool_call
        logger.debug("Turn %d: %s(%s)", self.state.turn, tool_name, arguments)

        result = self.tool_executor.execute(tool_name, arguments)
        feedback = format_tool_result(tool_name, result)
        self.state.add_message("observation", feedback)

        return feedback, False

    def get_conversation_for_generation(self, tokenizer: PreTrainedTokenizerBase | None = None) -> str:
        if self.state is None:
            raise RuntimeError("Call reset() before generating")
        tok = tokenizer or self.tokenizer
        if tok is None:
            raise ValueError("Tokenizer required for ChatML formatting (pass tokenizer= or set harness.tokenizer)")
        return messages_to_chatml(self.state.messages, tokenizer=tok, add_generation_prompt=True)
