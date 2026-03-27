#!/usr/bin/env python3
"""SWE-bench Lite evaluation for agent coding capability.

Evaluates a model checkpoint on SWE-bench Lite tasks by running
the full agent loop (tool use in sandboxed environments) and
measuring test pass rate.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opencomposer.checkpoint_utils import maybe_prune_before_hf_load
from opencomposer.agent.harness import AgentHarness
from opencomposer.agent.prompts import parse_tool_call
from opencomposer.environments.docker_env import BuiltinEnvironment
from opencomposer.environments.reward import RewardComputer
from opencomposer.data.rl_prompts import load_rl_tasks, RLTask

logger = logging.getLogger(__name__)


def run_agent_episode(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    task: RLTask,
    max_turns: int = 20,
    max_new_tokens: int = 1024,
) -> dict[str, Any]:
    """Run one agent episode on a task and return results."""
    env = BuiltinEnvironment(timeout=60)
    workspace = env.setup_from_task(task)

    harness = AgentHarness(workspace_dir=workspace, max_turns=max_turns)
    harness.reset(task.description)

    episode_log = []
    for turn in range(max_turns):
        prompt = harness.get_conversation_for_generation()
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=8192,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated, skip_special_tokens=True)

        feedback, done = harness.step(response)
        episode_log.append({
            "turn": turn + 1,
            "response_preview": response[:200],
            "done": done,
        })

        if done:
            break

    # Compute reward
    reward_computer = RewardComputer()
    reward, info = reward_computer.compute(env, task.test_command)

    env.teardown()

    return {
        "task_id": task.task_id,
        "reward": reward,
        "info": info,
        "turns": len(episode_log),
        "episode_log": episode_log,
    }


def evaluate_swebench_lite(
    model_path: str,
    task_source: str = "builtin",
    max_tasks: int | None = None,
    max_turns: int = 20,
) -> dict[str, Any]:
    """Run SWE-bench Lite evaluation.

    Returns:
        Dict with pass_rate, total, results per task.
    """
    logger.info("Loading model from %s", model_path)
    maybe_prune_before_hf_load(model_path, logger)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    tasks = load_rl_tasks(source=task_source)
    if max_tasks:
        tasks = tasks[:max_tasks]

    results = []
    passed = 0
    for task in tasks:
        logger.info("Running task: %s", task.task_id)
        result = run_agent_episode(model, tokenizer, task, max_turns=max_turns)
        results.append(result)

        if result["reward"] > 0.5:
            passed += 1
            logger.info("  %s: PASS (reward=%.2f, turns=%d)",
                        task.task_id, result["reward"], result["turns"])
        else:
            logger.info("  %s: FAIL (reward=%.2f, turns=%d)",
                        task.task_id, result["reward"], result["turns"])

    total = len(tasks)
    pass_rate = passed / total if total > 0 else 0.0

    return {
        "pass_rate": pass_rate,
        "passed": passed,
        "total": total,
        "results": results,
    }
