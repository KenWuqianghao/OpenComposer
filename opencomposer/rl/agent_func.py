"""OpenRLHF multi-turn agent implementation for coding RL training.

This module implements the AgentInstanceBase interface required by OpenRLHF's
multi-turn agent execution pipeline. Each instance manages one RL episode
where the model interacts with a sandboxed coding environment via tools.
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any, Dict

import torch

from openrlhf.utils.agent import AgentInstanceBase, MultiTurnAgentExecutor

from opencomposer.agent.harness import AgentHarness
from opencomposer.agent.prompts import format_tool_result, parse_tool_call
from opencomposer.environments.docker_env import BuiltinEnvironment
from opencomposer.environments.reward import RewardComputer
from opencomposer.data.rl_prompts import load_rl_tasks, RLTask
from opencomposer.self_summarization.summarizer import SelfSummarizer, SummarizationConfig

logger = logging.getLogger(__name__)

_TASK_POOL: list[RLTask] | None = None
_REWARD_COMPUTER: RewardComputer | None = None

MAX_TURNS = 30


def _get_task_pool() -> list[RLTask]:
    global _TASK_POOL
    if _TASK_POOL is None:
        _TASK_POOL = load_rl_tasks(source="builtin")
        logger.info("Loaded %d RL tasks into pool", len(_TASK_POOL))
    return _TASK_POOL


def _get_reward_computer() -> RewardComputer:
    global _REWARD_COMPUTER
    if _REWARD_COMPUTER is None:
        _REWARD_COMPUTER = RewardComputer(
            pass_reward=1.0,
            fail_reward=-0.5,
            partial_reward=True,
        )
    return _REWARD_COMPUTER


class AgentInstance(AgentInstanceBase):
    """Single RL episode managing agent-environment interaction."""

    def __init__(self, *args, **kwargs):
        self.step_idx = 0
        self.task: RLTask | None = None
        self.env: BuiltinEnvironment | None = None
        self.harness: AgentHarness | None = None
        self.summarizer = SelfSummarizer(SummarizationConfig(
            context_trigger_tokens=8192,
            summary_max_tokens=1024,
            min_turns_before_summary=3,
            max_summarizations_per_episode=5,
        ))

    async def reset(self, states: dict, **kwargs) -> dict:
        """Initialize a new episode: pick a task, set up environment."""
        self.step_idx = 0

        if self.env is not None:
            self.env.teardown()

        task_pool = _get_task_pool()
        self.task = random.choice(task_pool)

        self.env = BuiltinEnvironment(timeout=60)
        workspace = self.env.setup_from_task(self.task)

        self.harness = AgentHarness(
            workspace_dir=workspace,
            max_turns=MAX_TURNS,
        )
        self.harness.reset(self.task.description)
        self.summarizer.reset(self.task.description)

        observation = self.harness.get_conversation_for_generation()
        return {"observation": observation}

    async def step(self, states: dict, **kwargs) -> Dict[str, Any]:
        """Process one turn of agent interaction.

        The self-summarization mechanism works as follows:
        1. Track token count across turns
        2. When context exceeds the trigger, inject a summarization prompt
        3. The model generates a summary as its next action
        4. Context resets to the summary; episode continues
        5. The final reward (from test execution) propagates through the
           entire trajectory including summaries, reinforcing good ones.
        """
        self.step_idx += 1
        action_text = states.get("action_text", "")

        # Update token tracking in the summarizer
        estimated_tokens = int(len(action_text.split()) * 1.3)
        self.summarizer.update_token_count(estimated_tokens)

        # Process through the agent harness (tool execution)
        feedback, is_done = self.harness.step(action_text)

        # Check if self-summarization should trigger
        if self.summarizer.should_summarize() and not is_done:
            summary_prompt = self.summarizer.get_summarization_prompt()
            feedback = (feedback + summary_prompt) if feedback else summary_prompt

        if is_done or self.step_idx >= MAX_TURNS:
            reward_computer = _get_reward_computer()
            reward_value, info = reward_computer.compute(
                self.env,
                self.task.test_command,
            )
            reward = torch.tensor(reward_value, dtype=torch.float32)
            is_done = True

            self.env.teardown()
            self.env = None

            logger.debug(
                "Episode done: task=%s, turns=%d, reward=%.2f, reason=%s, summaries=%d",
                self.task.task_id, self.step_idx, reward_value,
                info.get("reason", ""),
                self.summarizer.state.num_prior_summarizations,
            )
        else:
            reward = torch.tensor(0.0, dtype=torch.float32)

        # Format feedback for the next generation step
        if feedback and not is_done:
            environment_feedback = feedback + "\n\n<|assistant|>\n"
        elif is_done:
            environment_feedback = "\n</s>"
        else:
            environment_feedback = "\n<|assistant|>\n"

        return {
            "rewards": reward,
            "scores": reward,
            "environment_feedback": environment_feedback,
            "done": is_done,
            "sampling_params": states.get("sampling_params", None),
            "extra_logs": {
                "step": self.step_idx,
                "task_id": self.task.task_id if self.task else "",
                "context_tokens": self.summarizer.state.total_tokens_generated,
                "num_summarizations": self.summarizer.state.num_prior_summarizations,
            },
        }


class AgentExecutor(MultiTurnAgentExecutor):
    """OpenRLHF-compatible executor wrapping our AgentInstance."""

    def __init__(self):
        super().__init__(AgentInstance)
