#!/usr/bin/env python3
"""HumanEval evaluation for code generation quality.

Evaluates a model checkpoint on the HumanEval benchmark to measure
basic code generation capability at each stage of the pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opencomposer.checkpoint_utils import maybe_prune_before_hf_load

logger = logging.getLogger(__name__)

# HumanEval problems (subset for quick evaluation)
HUMANEVAL_PROBLEMS = [
    {
        "task_id": "HumanEval/0",
        "prompt": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
        "test": "def check(candidate):\n    assert candidate([1.0, 2.0, 3.0], 0.5) == False\n    assert candidate([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n",
        "entry_point": "has_close_elements",
    },
    {
        "task_id": "HumanEval/1",
        "prompt": "from typing import List\n\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n",
        "test": "def check(candidate):\n    assert candidate('(()()) ((())) () ((())()())') == ['(()())', '((()))', '()', '((())()())']\n    assert candidate('() (()) ((())) (((())))') == ['()', '(())', '((()))', '(((())))']\n    assert candidate('(()(()))') == ['(()(()))']\n    assert candidate('( ) (( )) (( )( ))') == ['()', '(())', '(()())']\n",
        "entry_point": "separate_paren_groups",
    },
    {
        "task_id": "HumanEval/2",
        "prompt": "\n\ndef truncate_number(number: float) -> float:\n    \"\"\" Given a positive floating point number, it can be decomposed into\n    and integer part (largest integer smaller than given number) and decimals\n    (leftover part always smaller than 1).\n\n    Return the decimal part of the number.\n    >>> truncate_number(3.5)\n    0.5\n    \"\"\"\n",
        "test": "def check(candidate):\n    assert candidate(3.5) == 0.5\n    assert abs(candidate(1.33) - 0.33) < 1e-6\n    assert abs(candidate(123.456) - 0.456) < 1e-6\n",
        "entry_point": "truncate_number",
    },
    {
        "task_id": "HumanEval/4",
        "prompt": "from typing import List\n\n\ndef mean_absolute_deviation(numbers: List[float]) -> float:\n    \"\"\" For a given list of input numbers, calculate Mean Absolute Deviation\n    around the mean of this dataset.\n    Mean Absolute Deviation is the average absolute difference between each\n    element and a centerpoint (mean in this case):\n    MAD = average | x - x_mean |\n    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])\n    1.0\n    \"\"\"\n",
        "test": "def check(candidate):\n    assert abs(candidate([1.0, 2.0, 3.0]) - 2.0/3.0) < 1e-6\n    assert abs(candidate([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-6\n    assert abs(candidate([1.0, 2.0, 3.0, 4.0, 5.0]) - 6.0/5.0) < 1e-6\n",
        "entry_point": "mean_absolute_deviation",
    },
    {
        "task_id": "HumanEval/11",
        "prompt": "from typing import List\n\n\ndef string_xor(a: str, b: str) -> str:\n    \"\"\" Input are two strings a and b consisting only of 1s and 0s.\n    Perform binary XOR on these inputs and return result also as a string.\n    >>> string_xor('010', '110')\n    '100'\n    \"\"\"\n",
        "test": "def check(candidate):\n    assert candidate('111000', '101010') == '010010'\n    assert candidate('1', '1') == '0'\n    assert candidate('0101', '0000') == '0101'\n",
        "entry_point": "string_xor",
    },
    {
        "task_id": "HumanEval/17",
        "prompt": "from typing import List\n\n\ndef parse_music(music_string: str) -> List[int]:\n    \"\"\" Input to this function is a string representing musical notes in a special ASCII format.\n    Your task is to parse this string and return list of integers corresponding to how many beats does each\n    not last.\n\n    Here is a legend:\n    'o' - whole note, lasts four beats\n    'o|' - half note, lasts two beats\n    '.|' - quarter note, lasts one beat\n\n    >>> parse_music('o o| .| o| o| .| .| .| .| o o')\n    [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]\n    \"\"\"\n",
        "test": "def check(candidate):\n    assert candidate('') == []\n    assert candidate('o o o o') == [4, 4, 4, 4]\n    assert candidate('.| .| .| .|') == [1, 1, 1, 1]\n    assert candidate('o| o| .| .| o o o o') == [2, 2, 1, 1, 4, 4, 4, 4]\n    assert candidate('o| .| o| .| o o| o o|') == [2, 1, 2, 1, 4, 2, 4, 2]\n",
        "entry_point": "parse_music",
    },
    {
        "task_id": "HumanEval/26",
        "prompt": "from typing import List\n\n\ndef remove_duplicates(numbers: List[int]) -> List[int]:\n    \"\"\" From a list of integers, remove all elements that occur more than once.\n    Keep order of elements left the same as in the input.\n    >>> remove_duplicates([1, 2, 3, 2, 4])\n    [1, 3, 4]\n    \"\"\"\n",
        "test": "def check(candidate):\n    assert candidate([]) == []\n    assert candidate([1, 2, 3, 4]) == [1, 2, 3, 4]\n    assert candidate([1, 2, 3, 2, 4, 3, 5]) == [1, 4, 5]\n",
        "entry_point": "remove_duplicates",
    },
    {
        "task_id": "HumanEval/28",
        "prompt": "from typing import List\n\n\ndef concatenate(strings: List[str]) -> str:\n    \"\"\" Concatenate list of strings into a single string\n    >>> concatenate([])\n    ''\n    >>> concatenate(['a', 'b', 'c'])\n    'abc'\n    \"\"\"\n",
        "test": "def check(candidate):\n    assert candidate([]) == ''\n    assert candidate(['x', 'y', 'z']) == 'xyz'\n    assert candidate(['x', 'y', 'z', 'w', 'k']) == 'xyzwk'\n",
        "entry_point": "concatenate",
    },
]


def generate_completion(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """Generate a code completion for a HumanEval prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    completion = tokenizer.decode(generated, skip_special_tokens=True)

    # Stop at the first function definition or class that isn't part of the answer
    stop_sequences = ["\ndef ", "\nclass ", "\n# ", "\nif __name__"]
    for stop in stop_sequences:
        idx = completion.find(stop)
        if idx >= 0:
            completion = completion[:idx]

    return completion


def execute_test(prompt: str, completion: str, test: str, entry_point: str, timeout: int = 10) -> bool:
    """Execute a HumanEval test case and return pass/fail."""
    full_code = prompt + completion + "\n\n" + test + f"\ncheck({entry_point})\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        f.flush()
        try:
            result = subprocess.run(
                ["python3", f.name],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        finally:
            os.unlink(f.name)


def evaluate_humaneval(
    model_path: str,
    problems: list[dict] | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Run HumanEval evaluation on a model checkpoint.

    Returns:
        Dict with pass_rate, total, passed, and per-problem results.
    """
    problems = problems or HUMANEVAL_PROBLEMS

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

    results = []
    passed = 0
    for problem in problems:
        task_id = problem["task_id"]
        logger.info("Evaluating %s", task_id)

        completion = generate_completion(
            model, tokenizer, problem["prompt"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        success = execute_test(
            problem["prompt"], completion,
            problem["test"], problem["entry_point"],
        )

        results.append({
            "task_id": task_id,
            "completion": completion,
            "passed": success,
        })
        if success:
            passed += 1
        logger.info("  %s: %s", task_id, "PASS" if success else "FAIL")

    total = len(problems)
    pass_rate = passed / total if total > 0 else 0.0

    return {
        "pass_rate": pass_rate,
        "passed": passed,
        "total": total,
        "results": results,
    }
