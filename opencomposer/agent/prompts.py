"""System prompts and tool schema definitions (Qwen ChatML)."""

from __future__ import annotations

import json

from transformers import PreTrainedTokenizerBase

TOOL_SCHEMAS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file at the given path.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "File path relative to workspace root"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file, creating or overwriting it.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to workspace root"},
                "content": {"type": "string", "description": "Full file content to write"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace a unique occurrence of old_str with new_str in the file at path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_str": {"type": "string", "description": "Exact string to find (must be unique in the file)"},
                "new_str": {"type": "string", "description": "Replacement string"},
            },
            "required": ["path", "old_str", "new_str"],
        },
    },
    {
        "name": "grep",
        "description": "Search for a regex pattern in files under the given path.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for"},
                "path": {"type": "string", "description": "Directory or file to search in (default: '.')"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "run_terminal",
        "description": "Execute a shell command and return stdout/stderr.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "Shell command to execute"}},
            "required": ["command"],
        },
    },
    {
        "name": "search_codebase",
        "description": "Search the codebase for files related to a natural-language query.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Natural language search query"}},
            "required": ["query"],
        },
    },
]

SYSTEM_PROMPT = f"""\
You are a coding agent that solves software engineering tasks. You have access to these tools:

{json.dumps(TOOL_SCHEMAS, indent=2)}

To call a tool, output a <tool_call> block with the tool name and arguments as JSON:

<tool_call>
{{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}
</tool_call>

After each tool call, you will receive the result in an <observation> block. Use the result to inform your next action.

Guidelines:
- Read relevant files before making changes.
- Use grep or search_codebase to find code you need to understand.
- Make minimal, targeted edits.
- Run tests after making changes to verify correctness.
- When done, state that the task is complete.
"""

SUMMARIZATION_PROMPT = """\
Please summarize the current conversation context. Include:
1. The original task description
2. Key findings from reading/searching code
3. Changes made so far and their outcomes
4. Current status and remaining work
5. Any important details that must not be forgotten

Be concise but preserve all critical information needed to continue the task.\
"""


def format_system_message() -> str:
    return SYSTEM_PROMPT


def format_tool_result(tool_name: str, result: str) -> str:
    return f"<observation>\n[{tool_name} result]\n{result}\n</observation>"


def parse_tool_call(text: str) -> tuple[str, dict] | None:
    import re

    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    if not match:
        return None
    try:
        call = json.loads(match.group(1))
        name = call.get("name")
        arguments = call.get("arguments", {})
        if not name:
            return None
        return name, arguments
    except (json.JSONDecodeError, KeyError):
        return None


def messages_to_chatml(
    messages: list[dict[str, str]],
    *,
    tokenizer: PreTrainedTokenizerBase,
    add_generation_prompt: bool = True,
) -> str:
    """Build prompt string using the model's chat template (Qwen / compatible)."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    except Exception:
        # Fallback: minimal ChatML with eos as turn terminator
        im_start = "<|im_start|>"
        im_end = tokenizer.eos_token or ""
        parts: list[str] = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            parts.append(f"{im_start}{role}\n{content}{im_end}\n")
        if add_generation_prompt:
            parts.append(f"{im_start}assistant\n")
        return "".join(parts)
