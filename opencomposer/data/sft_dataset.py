"""Tool-use SFT dataset creation and loading (Stage 2).

Generates multi-turn conversations demonstrating the agent tool-calling format.
Can generate synthetic data from templates or load from existing datasets.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

from datasets import Dataset

logger = logging.getLogger(__name__)

TOOL_SCHEMAS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file at the given path.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    },
    {
        "name": "write_file",
        "description": "Write content to a file, creating or overwriting it.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace old_str with new_str in the file at path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_str": {"type": "string"},
                "new_str": {"type": "string"},
            },
            "required": ["path", "old_str", "new_str"],
        },
    },
    {
        "name": "grep",
        "description": "Search for a regex pattern in files under the given path.",
        "parameters": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
            "required": ["pattern"],
        },
    },
    {
        "name": "run_terminal",
        "description": "Execute a shell command and return stdout/stderr.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    },
    {
        "name": "search_codebase",
        "description": "Semantic keyword search across the codebase.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
]

# ---------------------------------------------------------------------------
# Synthetic SFT example templates
# ---------------------------------------------------------------------------

_BUG_FIX_TEMPLATES = [
    {
        "task": "Fix the TypeError in {file} where {var} is called as a function but is actually a string.",
        "tool_trace": [
            ("read_file", {"path": "{file}"}, "def main():\n    {var} = 'hello'\n    result = {var}()\n    return result"),
            ("edit_file", {"path": "{file}", "old_str": "result = {var}()", "new_str": "result = {var}"}, "OK"),
            ("run_terminal", {"command": "python -m pytest {test_file} -x"}, "1 passed"),
        ],
    },
    {
        "task": "Fix the IndexError in {file} that occurs when the input list is empty.",
        "tool_trace": [
            ("read_file", {"path": "{file}"}, "def process(items):\n    return items[0]"),
            ("edit_file", {"path": "{file}", "old_str": "return items[0]", "new_str": "return items[0] if items else None"}, "OK"),
            ("run_terminal", {"command": "python -m pytest {test_file} -x"}, "2 passed"),
        ],
    },
    {
        "task": "The function {func} in {file} returns incorrect results. Find and fix the off-by-one error.",
        "tool_trace": [
            ("grep", {"pattern": "def {func}", "path": "."}, "{file}:10: def {func}(n):"),
            ("read_file", {"path": "{file}"}, "def {func}(n):\n    return list(range(1, n))"),
            ("edit_file", {"path": "{file}", "old_str": "range(1, n)", "new_str": "range(1, n + 1)"}, "OK"),
            ("run_terminal", {"command": "python -m pytest {test_file} -x"}, "3 passed"),
        ],
    },
]

_FEATURE_TEMPLATES = [
    {
        "task": "Add a {func} function to {file} that {description}.",
        "tool_trace": [
            ("read_file", {"path": "{file}"}, "# existing module code\n\ndef existing_func():\n    pass"),
            ("search_codebase", {"query": "similar {func} implementation"}, "No results found."),
            ("edit_file", {"path": "{file}", "old_str": "def existing_func():\n    pass", "new_str": "def {func}({args}):\n    {body}\n\ndef existing_func():\n    pass"}, "OK"),
            ("run_terminal", {"command": "python -m pytest {test_file} -x"}, "4 passed"),
        ],
    },
]

_FILL_VARS = {
    "file": ["src/utils.py", "src/core.py", "lib/helpers.py", "app/main.py", "pkg/process.py"],
    "test_file": ["tests/test_utils.py", "tests/test_core.py", "tests/test_helpers.py"],
    "var": ["config", "handler", "processor", "formatter", "validator"],
    "func": ["calculate_total", "parse_input", "validate_schema", "merge_records", "format_output"],
    "description": [
        "computes the sum of all numeric values in a dict",
        "parses a CSV string into a list of dicts",
        "validates a JSON object against a schema",
    ],
    "args": ["data", "items, key=None", "records, strict=False"],
    "body": ["return sum(v for v in data.values() if isinstance(v, (int, float)))"],
}


def _fill_template(template: dict, seed: int) -> dict:
    """Fill a template with random variable choices."""
    rng = random.Random(seed)
    replacements = {k: rng.choice(v) for k, v in _FILL_VARS.items()}

    def _sub(s: str) -> str:
        for k, v in replacements.items():
            s = s.replace("{" + k + "}", v)
        return s

    task = _sub(template["task"])
    tool_trace = []
    for tool_name, args, result in template["tool_trace"]:
        filled_args = {k: _sub(v) if isinstance(v, str) else v for k, v in args.items()}
        tool_trace.append((tool_name, filled_args, _sub(result)))
    return {"task": task, "tool_trace": tool_trace}


def _trace_to_messages(task: str, tool_trace: list[tuple]) -> list[dict[str, str]]:
    """Convert a task + tool trace into a ChatML-style message list."""
    system_msg = (
        "You are a coding agent. You have access to these tools:\n"
        + json.dumps(TOOL_SCHEMAS, indent=2)
        + "\n\nTo call a tool, output a <tool_call> block. After receiving the result, continue reasoning."
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": system_msg}]
    messages.append({"role": "user", "content": task})

    for i, (tool_name, args, result) in enumerate(tool_trace):
        thinking = f"I need to use {tool_name} to proceed."
        if i == 0:
            thinking = f"Let me investigate this issue. I'll start by using {tool_name}."
        elif i == len(tool_trace) - 1 and tool_name == "run_terminal":
            thinking = "Let me verify the fix by running the tests."

        tool_call = json.dumps({"name": tool_name, "arguments": args})
        assistant_content = f"{thinking}\n\n<tool_call>\n{tool_call}\n</tool_call>"
        messages.append({"role": "assistant", "content": assistant_content})
        messages.append({"role": "tool", "content": f"[{tool_name} result]\n{result}"})

    messages.append({"role": "assistant", "content": "The issue has been fixed and all tests pass."})
    return messages


def generate_synthetic_sft_data(num_examples: int = 5000, seed: int = 42) -> list[dict]:
    """Generate synthetic tool-use SFT training examples."""
    rng = random.Random(seed)
    all_templates = _BUG_FIX_TEMPLATES + _FEATURE_TEMPLATES
    examples = []

    for i in range(num_examples):
        template = rng.choice(all_templates)
        filled = _fill_template(template, seed=seed + i)
        messages = _trace_to_messages(filled["task"], filled["tool_trace"])
        examples.append({"messages": messages, "id": f"synthetic_{i}"})

    logger.info("Generated %d synthetic SFT examples", len(examples))
    return examples


def save_sft_dataset(output_dir: str, num_examples: int = 5000, seed: int = 42) -> Path:
    """Generate and save the SFT dataset to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    examples = generate_synthetic_sft_data(num_examples=num_examples, seed=seed)
    ds = Dataset.from_list(examples)

    split_ds = ds.train_test_split(test_size=0.05, seed=seed)
    split_ds["train"].save_to_disk(str(output_path / "train"))
    split_ds["test"].save_to_disk(str(output_path / "test"))

    logger.info("Saved SFT dataset to %s (train=%d, test=%d)",
                output_path, len(split_ds["train"]), len(split_ds["test"]))
    return output_path


def load_sft_dataset(dataset_path: str, split: str = "train") -> Dataset:
    """Load a previously saved SFT dataset."""
    path = Path(dataset_path) / split
    if not path.exists():
        raise FileNotFoundError(f"SFT dataset not found at {path}")
    return Dataset.load_from_disk(str(path))
