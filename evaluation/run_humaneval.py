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
from opencomposer.train_runtime import hf_causal_lm_eval_kw

logger = logging.getLogger(__name__)

# Tool-use SFT models default to JSON/tools; force a pure-code contract for HumanEval.
_HUMANEVAL_SYSTEM = (
    "You are a Python programmer solving short coding puzzles. "
    "You must output ONLY the missing part of the given function: normal indented Python "
    "(usually starting with 4 spaces). "
    "Do not call tools, do not emit JSON, do not use XML tags like <completion>, "
    "do not say 'Let me', do not mention files or pytest. No markdown."
)

_HUMANEVAL_USER_HEADER = (
    "Complete the function body below. Output nothing before the first indented line of code.\n\n"
)

# Concrete pattern helps tool-tuned checkpoints switch back to code completion.
_HUMANEVAL_FEW_SHOT = (
    "Follow this pattern: output ONLY the missing lines inside the function (indented with 4 spaces).\n\n"
    "Example (different exercise):\n"
    "def double_it(x: int) -> int:\n"
    '    """Return twice x."""\n'
    "You reply with only:\n"
    "    return x * 2\n\n"
    "Your exercise:\n"
)


def _strip_tool_use_debris(text: str) -> str:
    """Remove tool-SFT boilerplate (JSON tools, fake XML, investigation prefaces)."""
    t = text.strip()
    t = re.sub(r"(?is)<completion>.*?</completion>", "", t)
    t = re.sub(r"(?is)<tool_call>.*?</tool_call>", "", t)
    lines_out: list[str] = []
    for line in t.split("\n"):
        s = line.strip()
        if not s:
            lines_out.append(line)
            continue
        if s.startswith("{") and '"name"' in s:
            continue
        # Synthetic tool/SFT transcripts often leak HTML-ish closing stubs; drop whole line.
        if re.fullmatch(r"</[a-zA-Z][\w.-]*>\s*", s):
            continue
        if s.startswith("Let me investigate") or s.startswith("Let me verify") or s.startswith("I'll start by"):
            continue
        lines_out.append(line)
    t = "\n".join(lines_out)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    # Prefer first block that looks like real function body lines.
    lines = t.split("\n")
    code_start = _first_humaneval_body_line_index(lines)
    if code_start is not None and code_start > 0:
        t = "\n".join(lines[code_start:]).strip()
    return t


_CODEISH = re.compile(
    r"^\s{4,}(return\b|if\b|for\b|while\b|elif\b|else:|pass\b|raise\b|"
    r"[a-zA-Z_][\w.]*\s*=|[a-zA-Z_][\w.]*\s*\(|async\s+def\b)",
)


def _first_humaneval_body_line_index(lines: list[str]) -> int | None:
    for i, line in enumerate(lines):
        if _CODEISH.match(line):
            return i
    return None


def _completion_looks_like_tool_dump(completion: str) -> bool:
    """True if the model stayed in agent/tool format instead of emitting Python."""
    if not completion.strip():
        return True
    t = completion.strip()
    if "<completion>" in t or "</completion>" in t.lower():
        return True
    if re.search(r"\{\s*[\"']name[\"']\s*:", t):
        return True
    if "Let me investigate" in t[:400] or "I'll start by using" in t[:400]:
        return True
    # Closing tags / synthetic helpers paths (tool transcripts), not Python.
    if "</" in t and _CODEISH.search(t) is None:
        return True
    if re.search(r"</[a-zA-Z]", t):
        return True
    if re.search(r"</(issue|prompt|task|section|p|div)\s*>", t, re.I):
        return True
    lines = [ln for ln in t.splitlines() if ln.strip()]
    if not lines:
        return True
    code_lines = sum(1 for ln in lines if _CODEISH.match(ln))
    if code_lines == 0 and len(t) > 20:
        return True
    return False


def _truncate_repeated_solution_lines(body: str) -> str:
    """Stop at consecutive duplicate logical lines (sampling loops on one wrong return)."""
    lines = body.split("\n")
    out: list[str] = []
    for ln in lines:
        if (
            out
            and ln.strip()
            and ln.strip() == out[-1].strip()
        ):
            break
        out.append(ln)
    return "\n".join(out)


def _strip_chat_code_completion(text: str) -> str:
    """Chat-tuned GLM often wraps code in ```python fences and may repeat blocks; keep one body."""
    text = text.strip()
    if "```" not in text:
        body = text
    else:
        blocks = re.findall(r"```(?:python)?\s*\n([\s\S]*?)```", text)
        if blocks:
            body = blocks[0].strip()
        elif text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            while lines and lines[-1].strip() == "```":
                lines.pop()
            body = "\n".join(lines).strip()
        else:
            body = text
    # HumanEval prompts end inside the function; body must be indented.
    lines = body.split("\n")
    if lines and lines[0].strip() and not lines[0].startswith((" ", "\t")):
        body = "\n".join(("    " + ln) if ln.strip() else ln for ln in lines)
    # Chat models often hallucinate a second exercise; cut before another top-level `def`.
    if "\n    def " in body:
        body = body.split("\n    def ", 1)[0].rstrip()
    body = _truncate_repeated_solution_lines(body)
    return body


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


def _is_chatglm_model(model: AutoModelForCausalLM) -> bool:
    arch = (getattr(model.config, "architectures", None) or [""])[0]
    return bool(arch and "ChatGLM" in arch)


def _chatglm_format_prompt(
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
    *,
    assistant_prefix: str | None = None,
) -> str:
    merged = messages[0]["content"]
    if len(messages) > 1:
        merged = messages[0]["content"] + "\n\n" + messages[1]["content"]
    if getattr(tokenizer, "chat_template", None):
        try:
            if assistant_prefix is not None:
                cont = tokenizer.apply_chat_template(
                    [*messages, {"role": "assistant", "content": assistant_prefix}],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                return cont
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            try:
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": merged}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
    return merged


def _generate_completion_chatglm_manual(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Token-by-token decode for ChatGLM (HF ``generate()`` + KV cache is buggy on this stack).

    Greedy argmax causes severe repetition/degeneration on GLM-4; use ``temperature > 0`` (default 0.25).
    Tool-use SFT checkpoints need a strict system prompt plus debris stripping; we retry once if needed.
    """
    device = next(model.parameters()).device
    base_temp = float(temperature) if temperature > 1e-6 else 0.42

    few_shot_turns = [
        {"role": "system", "content": _HUMANEVAL_SYSTEM},
        {"role": "user", "content": _HUMANEVAL_FEW_SHOT + prompt},
    ]
    simple_turns = [
        {"role": "system", "content": _HUMANEVAL_SYSTEM},
        {"role": "user", "content": _HUMANEVAL_USER_HEADER + prompt},
    ]

    # GLM tool-tuned checkpoints: anchor the assistant turn as code (no "Let me…" preamble).
    attempts: list[tuple[str | None, float]] = [
        (_chatglm_format_prompt(tokenizer, few_shot_turns, assistant_prefix="    "), base_temp),
        (_chatglm_format_prompt(tokenizer, few_shot_turns, assistant_prefix="    "), 0.08),
        (_chatglm_format_prompt(tokenizer, few_shot_turns), base_temp),
        (None, min(0.72, base_temp + 0.22)),
        (_chatglm_format_prompt(tokenizer, simple_turns), min(0.88, base_temp + 0.35)),
    ]

    eos = tokenizer.eos_token_id

    for formatted_text, temp in attempts:
        if formatted_text is None:
            enc = tokenizer(prompt, return_tensors="pt")
        else:
            enc = tokenizer(formatted_text, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        prompt_len = input_ids.shape[1]

        same_tok_run = 0
        prev_tok: int | None = None

        with torch.no_grad():
            for step in range(max_new_tokens):
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                logits = out.logits[:, -1, :].float()
                probs = torch.softmax(logits / temp, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                tid = int(next_token.item())
                if prev_tok == tid:
                    same_tok_run += 1
                else:
                    same_tok_run = 0
                    prev_tok = tid
                if same_tok_run > 28:
                    break

                input_ids = torch.cat([input_ids, next_token], dim=1)
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones_like(next_token, device=device)],
                        dim=1,
                    )
                if eos is not None and tid == eos:
                    break
                if step > 64 and step % 32 == 0:
                    partial = tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)
                    if partial.count("</") > 6 and _CODEISH.search(partial) is None:
                        break

        new_ids = input_ids[0, prompt_len:]
        raw = tokenizer.decode(new_ids, skip_special_tokens=True)
        completion = _strip_chat_code_completion(_strip_tool_use_debris(raw))
        if not _completion_looks_like_tool_dump(completion):
            return completion

    return ""


def _generate_completion_chat_template(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Qwen / ChatML and other models with ``tokenizer.chat_template`` (HumanEval contract)."""
    if os.environ.get("OPENCOMPOSER_MTP_SPECULATIVE", "").lower() in ("1", "true", "yes"):
        logger.info(
            "OPENCOMPOSER_MTP_SPECULATIVE is set but draft–target speculative decoding "
            "is not wired in HumanEval; using standard generate()."
        )

    device = next(model.parameters()).device
    few_shot_turns = [
        {"role": "system", "content": _HUMANEVAL_SYSTEM},
        {"role": "user", "content": _HUMANEVAL_FEW_SHOT + prompt},
    ]
    simple_turns = [
        {"role": "system", "content": _HUMANEVAL_SYSTEM},
        {"role": "user", "content": _HUMANEVAL_USER_HEADER + prompt},
    ]

    for messages in (few_shot_turns, simple_turns):
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            continue
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        prompt_len = inputs["input_ids"].shape[1]
        gen_inputs = {k: v for k, v in inputs.items() if k != "position_ids"}
        with torch.no_grad():
            outputs = model.generate(
                **gen_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )
        if hasattr(outputs, "sequences"):
            out_ids = outputs.sequences
        else:
            out_ids = outputs
        generated = out_ids[0, prompt_len:]
        raw = tokenizer.decode(generated, skip_special_tokens=True)
        completion = _strip_chat_code_completion(_strip_tool_use_debris(raw))
        if not _completion_looks_like_tool_dump(completion):
            return completion
    return ""


def _apply_humaneval_stop_sequences(completion: str) -> str:
    stop_sequences = ["\ndef ", "\nclass ", "\n# ", "\nif __name__"]
    for stop in stop_sequences:
        idx = completion.find(stop)
        if idx >= 0:
            completion = completion[:idx]
    return completion


def _generate_completion_raw_prefix(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    prompt_len = inputs["input_ids"].shape[1]
    gen_inputs = {k: v for k, v in inputs.items() if k != "position_ids"}

    with torch.no_grad():
        outputs = model.generate(
            **gen_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    if hasattr(outputs, "sequences"):
        out_ids = outputs.sequences
    else:
        out_ids = outputs
    generated = out_ids[0, prompt_len:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def generate_completion(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """Generate a code completion for a HumanEval prompt."""
    if _is_chatglm_model(model):
        completion = _generate_completion_chatglm_manual(
            model, tokenizer, prompt, max_new_tokens=max_new_tokens, temperature=temperature,
        )
    elif getattr(tokenizer, "chat_template", None):
        completion = _generate_completion_chat_template(
            model, tokenizer, prompt, max_new_tokens=max_new_tokens, temperature=temperature,
        )
        if not completion.strip():
            logger.warning("Chat-template HumanEval path returned empty; falling back to raw prompt.")
            completion = _generate_completion_raw_prefix(
                model, tokenizer, prompt, max_new_tokens=max_new_tokens, temperature=temperature,
            )
    else:
        completion = _generate_completion_raw_prefix(
            model, tokenizer, prompt, max_new_tokens=max_new_tokens, temperature=temperature,
        )

    return _apply_humaneval_stop_sequences(completion)


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
    hl = os.environ.get("OPENCOMPOSER_HUMANEVAL_LIMIT", "").strip()
    if hl.isdigit() and int(hl) > 0:
        problems = problems[: int(hl)]

    torch.manual_seed(42)

    logger.info("Loading model from %s", model_path)
    maybe_prune_before_hf_load(model_path, logger)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        **hf_causal_lm_eval_kw(),
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
