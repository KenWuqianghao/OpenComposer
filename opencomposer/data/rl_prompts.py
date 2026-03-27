"""RL task prompt loading for Stage 3.

Loads coding tasks for RL training — either from SWE-bench Lite or
from a set of built-in lightweight coding challenges.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RLTask:
    """A single RL training task (coding problem in a sandboxed environment)."""
    task_id: str
    description: str
    repo: str
    base_commit: str = ""
    test_command: str = "python -m pytest -x"
    setup_commands: list[str] = field(default_factory=list)
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Built-in lightweight tasks (no Docker required, for quick iteration)
# ---------------------------------------------------------------------------

BUILTIN_TASKS = [
    RLTask(
        task_id="builtin_001",
        description=(
            "Fix the function `fibonacci(n)` in `main.py` so that it returns the correct "
            "nth Fibonacci number. Currently it has an off-by-one error."
        ),
        repo="builtin",
        test_command="python -m pytest test_main.py -x",
        metadata={
            "files": {
                "main.py": (
                    "def fibonacci(n):\n"
                    "    if n <= 0:\n"
                    "        return 0\n"
                    "    if n == 1:\n"
                    "        return 1\n"
                    "    a, b = 0, 1\n"
                    "    for _ in range(n - 2):\n"  # bug: should be n - 1
                    "        a, b = b, a + b\n"
                    "    return b\n"
                ),
                "test_main.py": (
                    "from main import fibonacci\n\n"
                    "def test_fib_0():\n    assert fibonacci(0) == 0\n\n"
                    "def test_fib_1():\n    assert fibonacci(1) == 1\n\n"
                    "def test_fib_5():\n    assert fibonacci(5) == 5\n\n"
                    "def test_fib_10():\n    assert fibonacci(10) == 55\n\n"
                ),
            }
        },
    ),
    RLTask(
        task_id="builtin_002",
        description=(
            "The function `flatten(nested_list)` in `utils.py` should recursively flatten a "
            "nested list but currently raises a RecursionError on mixed types. Fix it."
        ),
        repo="builtin",
        test_command="python -m pytest test_utils.py -x",
        metadata={
            "files": {
                "utils.py": (
                    "def flatten(nested_list):\n"
                    "    result = []\n"
                    "    for item in nested_list:\n"
                    "        result.extend(flatten(item))\n"  # bug: no base case for non-list
                    "    return result\n"
                ),
                "test_utils.py": (
                    "from utils import flatten\n\n"
                    "def test_flat():\n    assert flatten([1, 2, 3]) == [1, 2, 3]\n\n"
                    "def test_nested():\n    assert flatten([[1, 2], [3, [4, 5]]]) == [1, 2, 3, 4, 5]\n\n"
                    "def test_empty():\n    assert flatten([]) == []\n\n"
                    "def test_mixed():\n    assert flatten([1, [2, 3], 4]) == [1, 2, 3, 4]\n\n"
                ),
            }
        },
    ),
    RLTask(
        task_id="builtin_003",
        description=(
            "Implement a `WordCounter` class in `counter.py` that counts word frequencies. "
            "The method `top_n(n)` should return the n most common words as a list of (word, count) tuples."
        ),
        repo="builtin",
        test_command="python -m pytest test_counter.py -x",
        metadata={
            "files": {
                "counter.py": (
                    "class WordCounter:\n"
                    "    def __init__(self):\n"
                    "        pass  # TODO: implement\n\n"
                    "    def add_text(self, text: str):\n"
                    "        pass  # TODO: implement\n\n"
                    "    def top_n(self, n: int):\n"
                    "        pass  # TODO: implement\n"
                ),
                "test_counter.py": (
                    "from counter import WordCounter\n\n"
                    "def test_basic():\n"
                    "    wc = WordCounter()\n"
                    "    wc.add_text('the cat sat on the mat')\n"
                    "    assert wc.top_n(1) == [('the', 2)]\n\n"
                    "def test_multiple_adds():\n"
                    "    wc = WordCounter()\n"
                    "    wc.add_text('hello world')\n"
                    "    wc.add_text('hello there')\n"
                    "    result = wc.top_n(2)\n"
                    "    assert result[0] == ('hello', 2)\n\n"
                    "def test_empty():\n"
                    "    wc = WordCounter()\n"
                    "    assert wc.top_n(5) == []\n\n"
                ),
            }
        },
    ),
    RLTask(
        task_id="builtin_004",
        description=(
            "Fix the `merge_sorted` function in `merge.py`. It should merge two sorted lists "
            "into one sorted list, but currently fails when one list is exhausted before the other."
        ),
        repo="builtin",
        test_command="python -m pytest test_merge.py -x",
        metadata={
            "files": {
                "merge.py": (
                    "def merge_sorted(a, b):\n"
                    "    result = []\n"
                    "    i, j = 0, 0\n"
                    "    while i < len(a) and j < len(b):\n"
                    "        if a[i] <= b[j]:\n"
                    "            result.append(a[i])\n"
                    "            i += 1\n"
                    "        else:\n"
                    "            result.append(b[j])\n"
                    "            j += 1\n"
                    "    return result\n"  # bug: doesn't append remaining elements
                ),
                "test_merge.py": (
                    "from merge import merge_sorted\n\n"
                    "def test_equal_length():\n"
                    "    assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]\n\n"
                    "def test_unequal():\n"
                    "    assert merge_sorted([1, 2], [3, 4, 5, 6]) == [1, 2, 3, 4, 5, 6]\n\n"
                    "def test_empty_first():\n"
                    "    assert merge_sorted([], [1, 2]) == [1, 2]\n\n"
                    "def test_both_empty():\n"
                    "    assert merge_sorted([], []) == []\n\n"
                ),
            }
        },
    ),
    RLTask(
        task_id="builtin_005",
        description=(
            "Add input validation to `parse_config` in `config.py`. It should raise ValueError "
            "for missing required keys ('host', 'port') and ensure 'port' is an integer."
        ),
        repo="builtin",
        test_command="python -m pytest test_config.py -x",
        metadata={
            "files": {
                "config.py": (
                    "def parse_config(raw: dict) -> dict:\n"
                    "    return {\n"
                    "        'host': raw['host'],\n"
                    "        'port': raw['port'],\n"
                    "        'debug': raw.get('debug', False),\n"
                    "    }\n"
                ),
                "test_config.py": (
                    "import pytest\n"
                    "from config import parse_config\n\n"
                    "def test_valid():\n"
                    "    cfg = parse_config({'host': 'localhost', 'port': 8080})\n"
                    "    assert cfg == {'host': 'localhost', 'port': 8080, 'debug': False}\n\n"
                    "def test_missing_host():\n"
                    "    with pytest.raises(ValueError):\n"
                    "        parse_config({'port': 8080})\n\n"
                    "def test_missing_port():\n"
                    "    with pytest.raises(ValueError):\n"
                    "        parse_config({'host': 'localhost'})\n\n"
                    "def test_port_not_int():\n"
                    "    with pytest.raises(ValueError):\n"
                    "        parse_config({'host': 'localhost', 'port': 'abc'})\n\n"
                ),
            }
        },
    ),
]


def load_swebench_tasks(split: str = "test") -> list[RLTask]:
    """Load SWE-bench Lite tasks from HuggingFace."""
    try:
        from datasets import load_dataset
        ds = load_dataset("SWE-bench/SWE-bench_Lite", split=split)
    except Exception as e:
        logger.warning("Could not load SWE-bench Lite: %s. Falling back to builtin tasks.", e)
        return BUILTIN_TASKS

    tasks = []
    for row in ds:
        tasks.append(RLTask(
            task_id=row["instance_id"],
            description=row["problem_statement"],
            repo=row["repo"],
            base_commit=row["base_commit"],
            test_command=f"python -m pytest {' '.join(json.loads(row['FAIL_TO_PASS']))} -x"
            if row.get("FAIL_TO_PASS") else "python -m pytest -x",
            fail_to_pass=json.loads(row["FAIL_TO_PASS"]) if row.get("FAIL_TO_PASS") else [],
            pass_to_pass=json.loads(row["PASS_TO_PASS"]) if row.get("PASS_TO_PASS") else [],
            metadata={"version": row.get("version", ""), "environment_setup_commit": row.get("environment_setup_commit", "")},
        ))
    logger.info("Loaded %d SWE-bench Lite tasks", len(tasks))
    return tasks


def load_builtin_tasks() -> list[RLTask]:
    """Return the built-in lightweight coding tasks."""
    return list(BUILTIN_TASKS)


def load_rl_tasks(source: str = "builtin") -> list[RLTask]:
    """Load RL tasks from the specified source."""
    if source == "builtin":
        return load_builtin_tasks()
    elif source == "swebench":
        return load_swebench_tasks()
    else:
        raise ValueError(f"Unknown task source: {source}")
