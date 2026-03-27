"""Tool implementations for the coding agent.

Each tool operates within a sandbox workspace directory and returns
string results suitable for inclusion in the model's context.
"""

from __future__ import annotations

import os
import re
import subprocess
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_OUTPUT_LENGTH = 10000
COMMAND_TIMEOUT = 30


class ToolError(Exception):
    """Raised when a tool execution fails."""
    pass


class ToolExecutor:
    """Executes agent tools within a sandboxed workspace."""

    def __init__(self, workspace_dir: str, timeout: int = COMMAND_TIMEOUT):
        self.workspace = Path(workspace_dir).resolve()
        self.timeout = timeout
        if not self.workspace.exists():
            self.workspace.mkdir(parents=True, exist_ok=True)

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Dispatch and execute a tool call, returning the string result."""
        dispatch = {
            "read_file": self._read_file,
            "write_file": self._write_file,
            "edit_file": self._edit_file,
            "grep": self._grep,
            "run_terminal": self._run_terminal,
            "search_codebase": self._search_codebase,
        }
        handler = dispatch.get(tool_name)
        if handler is None:
            return f"Error: Unknown tool '{tool_name}'. Available tools: {list(dispatch.keys())}"
        try:
            return handler(**arguments)
        except ToolError as e:
            return f"Error: {e}"
        except Exception as e:
            logger.warning("Tool %s failed: %s", tool_name, e)
            return f"Error executing {tool_name}: {type(e).__name__}: {e}"

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to workspace, preventing directory traversal."""
        resolved = (self.workspace / path).resolve()
        if not str(resolved).startswith(str(self.workspace)):
            raise ToolError(f"Path '{path}' is outside the workspace")
        return resolved

    def _read_file(self, path: str) -> str:
        target = self._resolve_path(path)
        if not target.exists():
            return f"Error: File not found: {path}"
        if not target.is_file():
            return f"Error: Not a file: {path}"
        try:
            content = target.read_text(errors="replace")
        except Exception as e:
            return f"Error reading file: {e}"
        if len(content) > MAX_OUTPUT_LENGTH:
            content = content[:MAX_OUTPUT_LENGTH] + f"\n... (truncated, {len(content)} total chars)"
        return content

    def _write_file(self, path: str, content: str) -> str:
        target = self._resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"Successfully wrote {len(content)} chars to {path}"

    def _edit_file(self, path: str, old_str: str, new_str: str) -> str:
        target = self._resolve_path(path)
        if not target.exists():
            return f"Error: File not found: {path}"
        content = target.read_text(errors="replace")
        count = content.count(old_str)
        if count == 0:
            return f"Error: old_str not found in {path}"
        if count > 1:
            return f"Error: old_str found {count} times in {path} (must be unique)"
        new_content = content.replace(old_str, new_str, 1)
        target.write_text(new_content)
        return "OK"

    def _grep(self, pattern: str, path: str = ".") -> str:
        target = self._resolve_path(path)
        if not target.exists():
            return f"Error: Path not found: {path}"

        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Error: Invalid regex pattern: {e}"

        results = []
        search_root = target if target.is_dir() else target.parent
        files = [target] if target.is_file() else sorted(search_root.rglob("*"))

        for fpath in files:
            if not fpath.is_file():
                continue
            if fpath.suffix in (".pyc", ".so", ".o", ".exe", ".bin"):
                continue
            try:
                lines = fpath.read_text(errors="replace").splitlines()
                for i, line in enumerate(lines, 1):
                    if regex.search(line):
                        rel = fpath.relative_to(self.workspace)
                        results.append(f"{rel}:{i}: {line.rstrip()}")
            except Exception:
                continue
            if len(results) >= 100:
                results.append("... (truncated at 100 matches)")
                break

        if not results:
            return "No matches found."
        return "\n".join(results)

    def _run_terminal(self, command: str) -> str:
        # Basic command sanitization
        dangerous = ["rm -rf /", "mkfs", "dd if=", ": (){ :|:& };:"]
        for d in dangerous:
            if d in command:
                return f"Error: Potentially dangerous command blocked: {command}"

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += ("\n" if output else "") + result.stderr
            if not output:
                output = f"(exit code: {result.returncode})"
            else:
                output += f"\n(exit code: {result.returncode})"

            if len(output) > MAX_OUTPUT_LENGTH:
                output = output[:MAX_OUTPUT_LENGTH] + "\n... (output truncated)"
            return output
        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {self.timeout}s"

    def _search_codebase(self, query: str) -> str:
        """Simple keyword-based codebase search (semantic search stub)."""
        keywords = query.lower().split()
        results = []

        for fpath in sorted(self.workspace.rglob("*.py")):
            if not fpath.is_file():
                continue
            try:
                content = fpath.read_text(errors="replace")
                content_lower = content.lower()
                score = sum(1 for kw in keywords if kw in content_lower)
                if score > 0:
                    rel = fpath.relative_to(self.workspace)
                    # Extract first matching line for context
                    for line in content.splitlines():
                        if any(kw in line.lower() for kw in keywords):
                            results.append((score, str(rel), line.strip()))
                            break
            except Exception:
                continue

        if not results:
            return "No results found."

        results.sort(key=lambda x: -x[0])
        lines = []
        for score, path, line in results[:20]:
            lines.append(f"{path}: {line}")
        return "\n".join(lines)
