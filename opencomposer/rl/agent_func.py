"""OpenRLHF multi-turn agent implementation for coding RL training."""

from __future__ import annotations

import logging
import os
import random
import re
from typing import Any, Dict

import torch

try:
    from openrlhf.utils.agent import AgentInstanceBase, MultiTurnAgentExecutor
    _OPENRLHF_AVAILABLE = True
except Exception:
    _OPENRLHF_AVAILABLE = False

    class AgentInstanceBase:  # type: ignore[no-redef]
        pass

    class MultiTurnAgentExecutor:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("OpenRLHF is required for Stage 3 training.")

from opencomposer.agent.harness import AgentHarness
from opencomposer.data.rl_prompts import load_rl_tasks, RLTask
from opencomposer.environments.docker_env import BuiltinEnvironment
from opencomposer.environments.reward import RewardComputer
from opencomposer.rl.behavior_rewards import behavior_aux_score
from opencomposer.rl.length_penalty import episode_length_penalty_reward_delta
from opencomposer.self_summarization.summarizer import SelfSummarizer, SummarizationConfig

logger = logging.getLogger(__name__)

_TASK_POOL: list[RLTask] | None = None
_REWARD_COMPUTER: RewardComputer | None = None

MAX_TURNS = 30


def _load_tokenizer():
    for key in ("OPENCOMPOSER_MODEL_PATH", "PRETRAIN_PATH"):
        p = os.environ.get(key, "")
        if not p:
            continue
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(p, trust_remote_code=True)
        except Exception as e:
            logger.warning("Could not load tokenizer from %s=%s: %s", key, p, e)
    return None


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


def _count_think_tokens(text: str) -> int:
    total = 0
    for pat in (r"<think>([\s\S]*?)</think>", r"<thinking>([\s\S]*?)</thinking>"):
        for m in re.finditer(pat, text, re.I):
            total += len(m.group(1).split())
    return total


def _episode_stats_from_transcript(transcript: str, steps: int, tool_calls: int) -> dict:
    tool_call_tokens = len(re.findall(r"<tool_call>", transcript)) * 8
    obs_tokens = sum(
        len(m.group(1).split()) for m in re.finditer(r"<observation>([\s\S]*?)</observation>", transcript)
    )
    fc = transcript.lower().find("<|im_start|>assistant")
    tail = transcript[fc:] if fc >= 0 else transcript
    final_seg = tail[-1500:] if len(tail) > 1500 else tail
    return {
        "thinking_tokens": float(_count_think_tokens(transcript)),
        "tool_call_tokens": float(tool_call_tokens),
        "tool_output_tokens": float(obs_tokens),
        "final_message_tokens": float(len(final_seg.split())),
        "num_tool_calls": float(tool_calls),
        "num_turns": float(steps),
    }


class AgentInstance(AgentInstanceBase):
    """Single RL episode managing agent-environment interaction."""

    def __init__(self, *args, **kwargs):
        self.step_idx = 0
        self.task: RLTask | None = None
        self.env: BuiltinEnvironment | None = None
        self.harness: AgentHarness | None = None
        self._tokenizer = _load_tokenizer()
        self.summarizer = SelfSummarizer(
            SummarizationConfig(
                context_trigger_tokens=int(os.environ.get("OPENCOMPOSER_SUMMARY_TRIGGER", "8192")),
                summary_max_tokens=int(os.environ.get("OPENCOMPOSER_SUMMARY_MAX", "1024")),
                min_turns_before_summary=3,
                max_summarizations_per_episode=5,
            ),
            tokenizer=self._tokenizer,
        )
        self._tool_calls = 0

    async def reset(self, states: dict, **kwargs) -> dict:
        self.step_idx = 0
        self._tool_calls = 0

        if self.env is not None:
            self.env.teardown()

        task_pool = _get_task_pool()
        self.task = random.choice(task_pool)

        self.env = BuiltinEnvironment(timeout=60)
        workspace = self.env.setup_from_task(self.task)

        self.harness = AgentHarness(
            workspace_dir=workspace,
            max_turns=MAX_TURNS,
            tokenizer=self._tokenizer,
        )
        self.harness.reset(self.task.description)
        self.summarizer.reset(self.task.description)

        if self._tokenizer is None:
            raise RuntimeError(
                "Tokenizer required for ChatML agent. Set OPENCOMPOSER_MODEL_PATH or PRETRAIN_PATH "
                "to your HF checkpoint id or directory before launching Ray."
            )
        observation = self.harness.get_conversation_for_generation(self._tokenizer)
        return {"observation": observation}

    async def step(self, states: dict, **kwargs) -> Dict[str, Any]:
        self.step_idx += 1
        action_text = states.get("action_text", "")

        estimated_tokens = int(len(action_text.split()) * 1.3)
        self.summarizer.update_token_count(estimated_tokens)

        assert self.harness is not None and self.env is not None and self.task is not None
        feedback, is_done = self.harness.step(action_text)

        if "<tool_call>" in action_text:
            self._tool_calls += 1

        if self.summarizer.should_summarize() and not is_done:
            summary_prompt = self.summarizer.get_summarization_prompt()
            feedback = (feedback + summary_prompt) if feedback else summary_prompt

        if is_done or self.step_idx >= MAX_TURNS:
            reward_computer = _get_reward_computer()
            reward_value, info = reward_computer.compute(
                self.env,
                self.task.test_command,
            )

            transcript = ""
            if self.harness.state:
                transcript = self.harness.state.full_transcript
            stats = _episode_stats_from_transcript(transcript, self.step_idx, self._tool_calls)

            k = float(os.environ.get("OPENCOMPOSER_LENGTH_K", "0.02"))
            q = float(os.environ.get("OPENCOMPOSER_LENGTH_Q", "1.5"))
            scale = float(os.environ.get("OPENCOMPOSER_LENGTH_SCALE", "0.02"))
            if os.environ.get("OPENCOMPOSER_LENGTH_PENALTY", "1") not in ("0", "false", "False"):
                reward_value += episode_length_penalty_reward_delta(stats, k=k, q=q, scale=scale)

            aux_raw = os.environ.get("OPENCOMPOSER_AUX_REWARDS", "style,comms,tool_quality")
            if aux_raw and aux_raw not in ("0", "none", "false", "False"):
                enabled = tuple(x.strip() for x in aux_raw.split(",") if x.strip())
                reward_value += behavior_aux_score(transcript, enabled=enabled)  # type: ignore[arg-type]

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

        eos = ""
        if self._tokenizer is not None and getattr(self._tokenizer, "eos_token", None):
            eos = self._tokenizer.eos_token or ""

        if feedback and not is_done:
            environment_feedback = feedback + "\n\n"
        elif is_done:
            environment_feedback = eos if eos else "\n"
        else:
            environment_feedback = ""

        if not is_done and self._tokenizer is not None:
            try:
                environment_feedback += self._tokenizer.apply_chat_template(
                    [],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                environment_feedback += "<|im_start|>assistant\n"

        extra_logs: dict[str, Any] = {
            "step": float(self.step_idx),
            "context_tokens": float(self.summarizer.state.total_tokens_generated),
            "num_summarizations": float(self.summarizer.state.num_prior_summarizations),
        }
        # OpenRLHF converts every extra_logs value to torch.tensor — must be numeric (no None).
        # When wiring router replay, add e.g. expert_trace_num_layers as a scalar count instead.
        return {
            "rewards": reward,
            "scores": reward,
            "environment_feedback": environment_feedback,
            "done": is_done,
            "sampling_params": states.get("sampling_params", None),
            "extra_logs": extra_logs,
        }


class AgentExecutor(MultiTurnAgentExecutor):
    """OpenRLHF-compatible executor wrapping our AgentInstance."""

    def __init__(self):
        if not _OPENRLHF_AVAILABLE:
            raise RuntimeError(
                "OpenRLHF is not installed. Install it to run Stage 3 RL training, "
                "or run `bash scripts/rl_training.sh` without OpenRLHF for a load+generate smoke only."
            )
        super().__init__(AgentInstance)
