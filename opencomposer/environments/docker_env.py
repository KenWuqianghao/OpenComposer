"""Docker-based sandboxed coding environment for RL training.

Manages the lifecycle of Docker containers that serve as isolated
workspaces for the agent during RL episodes.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SandboxEnvironment:
    """A sandboxed coding environment using either Docker or local temp directories.

    For the miniature pipeline, we support both:
    - Docker containers (production-like, more isolated)
    - Local temp directories (faster, no Docker overhead, good for builtin tasks)
    """

    def __init__(
        self,
        use_docker: bool = False,
        docker_image: str = "opencomposer/sandbox:latest",
        timeout: int = 300,
    ):
        self.use_docker = use_docker
        self.docker_image = docker_image
        self.timeout = timeout
        self._workspace_dir: str | None = None
        self._container = None

    @property
    def workspace_dir(self) -> str:
        if self._workspace_dir is None:
            raise RuntimeError("Environment not started. Call setup() first.")
        return self._workspace_dir

    def setup(self, files: dict[str, str] | None = None, repo_url: str = "", base_commit: str = "") -> str:
        """Create the sandbox environment and populate it with files.

        Args:
            files: Dict mapping relative paths to file contents (for builtin tasks)
            repo_url: Git repo URL to clone (for SWE-bench tasks)
            base_commit: Git commit to checkout after cloning

        Returns:
            Path to the workspace directory.
        """
        if self.use_docker:
            return self._setup_docker(files, repo_url, base_commit)
        return self._setup_local(files, repo_url, base_commit)

    def _setup_local(self, files: dict[str, str] | None, repo_url: str, base_commit: str) -> str:
        """Create a local temp directory as the sandbox."""
        self._workspace_dir = tempfile.mkdtemp(prefix="opencomposer_env_")
        workspace = Path(self._workspace_dir)

        if files:
            for rel_path, content in files.items():
                fpath = workspace / rel_path
                fpath.parent.mkdir(parents=True, exist_ok=True)
                fpath.write_text(content)
            logger.debug("Created local sandbox with %d files at %s", len(files), self._workspace_dir)
        elif repo_url:
            import subprocess
            subprocess.run(
                ["git", "clone", "--depth=1", repo_url, str(workspace / "repo")],
                check=True, capture_output=True, timeout=120,
            )
            if base_commit:
                subprocess.run(
                    ["git", "checkout", base_commit],
                    cwd=str(workspace / "repo"),
                    check=True, capture_output=True, timeout=30,
                )
            self._workspace_dir = str(workspace / "repo")

        return self._workspace_dir

    def _setup_docker(self, files: dict[str, str] | None, repo_url: str, base_commit: str) -> str:
        """Create a Docker container as the sandbox."""
        try:
            import docker as docker_lib
            client = docker_lib.from_env()
        except Exception as e:
            logger.warning("Docker not available (%s), falling back to local sandbox", e)
            self.use_docker = False
            return self._setup_local(files, repo_url, base_commit)

        # Create a temp dir to mount into the container
        self._workspace_dir = tempfile.mkdtemp(prefix="opencomposer_docker_")
        workspace = Path(self._workspace_dir)

        if files:
            for rel_path, content in files.items():
                fpath = workspace / rel_path
                fpath.parent.mkdir(parents=True, exist_ok=True)
                fpath.write_text(content)

        try:
            self._container = client.containers.run(
                self.docker_image,
                command="sleep infinity",
                volumes={self._workspace_dir: {"bind": "/workspace", "mode": "rw"}},
                detach=True,
                mem_limit="2g",
                cpu_period=100000,
                cpu_quota=100000,
                network_mode="none",
                working_dir="/workspace",
            )
            logger.debug("Started Docker container %s", self._container.short_id)
        except Exception as e:
            logger.warning("Failed to start Docker container (%s), falling back to local", e)
            self.use_docker = False

        return self._workspace_dir

    def run_command(self, command: str) -> tuple[str, int]:
        """Run a command in the sandbox and return (output, exit_code)."""
        if self.use_docker and self._container:
            return self._run_docker_command(command)
        return self._run_local_command(command)

    def _run_local_command(self, command: str) -> tuple[str, int]:
        import subprocess
        try:
            result = subprocess.run(
                command, shell=True, cwd=self._workspace_dir,
                capture_output=True, text=True, timeout=self.timeout,
            )
            output = result.stdout + result.stderr
            return output, result.returncode
        except subprocess.TimeoutExpired:
            return f"Command timed out after {self.timeout}s", 1

    def _run_docker_command(self, command: str) -> tuple[str, int]:
        try:
            exit_code, output = self._container.exec_run(
                ["bash", "-c", command],
                workdir="/workspace",
                demux=True,
            )
            stdout = (output[0] or b"").decode(errors="replace")
            stderr = (output[1] or b"").decode(errors="replace")
            return stdout + stderr, exit_code
        except Exception as e:
            return f"Docker exec error: {e}", 1

    def teardown(self):
        """Clean up the sandbox."""
        if self._container:
            try:
                self._container.stop(timeout=5)
                self._container.remove(force=True)
            except Exception:
                pass
            self._container = None

        if self._workspace_dir and os.path.exists(self._workspace_dir):
            try:
                shutil.rmtree(self._workspace_dir, ignore_errors=True)
            except Exception:
                pass
            self._workspace_dir = None

    def __del__(self):
        self.teardown()


class BuiltinEnvironment(SandboxEnvironment):
    """Environment for builtin lightweight tasks (no Docker needed)."""

    def __init__(self, timeout: int = 60):
        super().__init__(use_docker=False, timeout=timeout)

    def setup_from_task(self, task) -> str:
        """Set up workspace from an RLTask's metadata files."""
        files = task.metadata.get("files", {})
        return self.setup(files=files)
