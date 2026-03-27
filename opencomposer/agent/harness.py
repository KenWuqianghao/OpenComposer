"""Agent harness: orchestrates the model's interaction with tools.

Manages the multi-turn loop where the model generates actions (tool calls),
the tools execute in the sandbox, and results are fed back.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from opencomposer.agent.tools import ToolExecutor
from opencomposer.agent.prompts import (
    format_system_message,
    format_tool_result,
    parse_tool_call,
    SUMMARIZATION_PROMPT,
)

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """Tracks the state of a single agent episode."""
    task_description: str
    messages: list[dict[str, str]] = field(default_factory=list)
    turn: int = 0
    done: bool = False
    total_tokens: int = 0
    num_summarizations: int = 0

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def get_conversation_text(self) -> str:
        """Return the full conversation as a flat string."""
        parts = []
        for msg in self.messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"<|system|>\n{content}")
            elif role == "user":
                parts.append(f"<|user|>\n{content}")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}")
            elif role == "observation":
                parts.append(f"<|observation|>\n{content}")
        return "\n".join(parts) + "\n<|assistant|>\n"


class AgentHarness:
    """Manages one episode of agent-environment interaction."""

    def __init__(
        self,
        workspace_dir: str,
        max_turns: int = 30,
        command_timeout: int = 30,
    ):
        self.tool_executor = ToolExecutor(workspace_dir, timeout=command_timeout)
        self.max_turns = max_turns
        self.state: AgentState | None = None

    def reset(self, task_description: str) -> AgentState:
        """Initialize a new episode with a task description."""
        self.state = AgentState(task_description=task_description)
        self.state.add_message("system", format_system_message())
        self.state.add_message("user", task_description)
        return self.state

    def step(self, model_output: str) -> tuple[str, bool]:
        """Process one turn: parse model output, execute tools, return feedback.

        Returns:
            (feedback_text, is_done): The observation text and whether episode ended.
        """
        if self.state is None:
            raise RuntimeError("Call reset() before step()")

        self.state.turn += 1
        self.state.add_message("assistant", model_output)

        # Check for completion signals
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

        # Check max turns
        if self.state.turn >= self.max_turns:
            self.state.done = True
            return "Maximum turns reached. Episode ending.", True

        # Parse and execute tool call
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

    def get_conversation_for_generation(self) -> str:
        """Return current conversation formatted for model input."""
        if self.state is None:
            raise RuntimeError("Call reset() before generating")
        return self.state.get_conversation_text()
