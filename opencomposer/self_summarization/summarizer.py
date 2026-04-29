"""Self-summarization module for context compression during long episodes."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from transformers import PreTrainedTokenizerBase

from opencomposer.agent.prompts import messages_to_chatml

logger = logging.getLogger(__name__)


@dataclass
class SummarizationConfig:
    context_trigger_tokens: int = 8192
    summary_max_tokens: int = 1024
    min_turns_before_summary: int = 3
    max_summarizations_per_episode: int = 5
    preserve_system_prompt: bool = True
    preserve_last_n_turns: int = 2


@dataclass
class ConversationState:
    original_task: str = ""
    current_plan: str = ""
    changes_made: list[str] = field(default_factory=list)
    remaining_work: str = ""
    num_prior_summarizations: int = 0
    total_tokens_generated: int = 0
    turns_since_last_summary: int = 0


SUMMARIZATION_SYSTEM_PROMPT = """\
You must now summarize the current conversation. This is critical for continuing the task effectively.

Your summary MUST include:
1. TASK: The original task description (verbatim if possible)
2. FINDINGS: Key findings from reading and searching code
3. CHANGES: All changes made so far (file paths, what was changed, and outcomes)
4. STATUS: Current status (what's working, what's broken)
5. PLAN: Remaining work needed to complete the task
6. CRITICAL DETAILS: File paths, function names, variable names, error messages, or test results that must not be forgotten

Be concise but complete. Missing a critical detail could cause the task to fail.\
"""


class SelfSummarizer:
    def __init__(
        self,
        config: SummarizationConfig | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ):
        self.config = config or SummarizationConfig()
        self.tokenizer = tokenizer
        self._state = ConversationState()

    def reset(self, task_description: str):
        self._state = ConversationState(original_task=task_description)

    def update_token_count(self, new_tokens: int):
        self._state.total_tokens_generated += new_tokens
        self._state.turns_since_last_summary += 1

    def should_summarize(self) -> bool:
        if self._state.turns_since_last_summary < self.config.min_turns_before_summary:
            return False
        if self._state.num_prior_summarizations >= self.config.max_summarizations_per_episode:
            return False
        return self._state.total_tokens_generated >= self.config.context_trigger_tokens

    def get_summarization_prompt(self) -> str:
        self._state.num_prior_summarizations += 1
        summary_num = self._state.num_prior_summarizations

        prompt = (
            f"\n[CONTEXT LIMIT REACHED — Self-summarization #{summary_num}]\n\n"
            f"{SUMMARIZATION_SYSTEM_PROMPT}\n\n"
            f"Conversation state:\n"
            f"- Summarization number: {summary_num}\n"
            f"- Tokens generated so far: ~{self._state.total_tokens_generated}\n"
            f"- Turns since last summary: {self._state.turns_since_last_summary}\n"
        )
        return prompt

    def process_summary(self, summary_text: str) -> str:
        self._state.total_tokens_generated = len(summary_text.split()) * 2
        self._state.turns_since_last_summary = 0

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a coding agent continuing a task. Below is your self-summary "
                    f"from the prior context (summarization #{self._state.num_prior_summarizations})."
                ),
            },
            {
                "role": "user",
                "content": (
                    "[CONTINUED FROM SELF-SUMMARY]\n\n"
                    f"{summary_text}\n\n"
                    "Continue working on the task. Use your tools to make progress."
                ),
            },
        ]
        if self.tokenizer is not None:
            return messages_to_chatml(messages, tokenizer=self.tokenizer, add_generation_prompt=True)
        im_start = "<|im_start|>"
        im_end = ""
        parts = [f"{im_start}{m['role']}\n{m['content']}{im_end}\n" for m in messages]
        parts.append(f"{im_start}assistant\n")
        return "".join(parts)

    def format_for_training(
        self,
        pre_summary_tokens: list[int],
        summary_tokens: list[int],
        post_summary_tokens: list[int],
    ) -> dict[str, Any]:
        full_sequence = pre_summary_tokens + summary_tokens + post_summary_tokens

        token_types = (
            ["pre_summary"] * len(pre_summary_tokens)
            + ["summary"] * len(summary_tokens)
            + ["post_summary"] * len(post_summary_tokens)
        )

        return {
            "input_ids": full_sequence,
            "token_types": token_types,
            "num_summarizations": self._state.num_prior_summarizations,
            "summary_token_count": len(summary_tokens),
        }

    @property
    def state(self) -> ConversationState:
        return self._state
