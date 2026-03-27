"""SWE-bench environment adapter.

Wraps SWE-bench Lite tasks into the sandbox environment format
for RL training on real-world software engineering problems.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from opencomposer.environments.docker_env import SandboxEnvironment
from opencomposer.data.rl_prompts import RLTask

logger = logging.getLogger(__name__)

SWEBENCH_REPOS = {
    "astropy/astropy": "https://github.com/astropy/astropy.git",
    "django/django": "https://github.com/django/django.git",
    "matplotlib/matplotlib": "https://github.com/matplotlib/matplotlib.git",
    "pallets/flask": "https://github.com/pallets/flask.git",
    "psf/requests": "https://github.com/psf/requests.git",
    "pydata/xarray": "https://github.com/pydata/xarray.git",
    "pylint-dev/pylint": "https://github.com/pylint-dev/pylint.git",
    "pytest-dev/pytest": "https://github.com/pytest-dev/pytest.git",
    "scikit-learn/scikit-learn": "https://github.com/scikit-learn/scikit-learn.git",
    "sphinx-doc/sphinx": "https://github.com/sphinx-doc/sphinx.git",
    "sympy/sympy": "https://github.com/sympy/sympy.git",
}


class SWEBenchEnvironment:
    """Manages SWE-bench task environments for RL training."""

    def __init__(
        self,
        use_docker: bool = True,
        timeout: int = 300,
        cache_dir: str = "/tmp/swebench_repos",
    ):
        self.use_docker = use_docker
        self.timeout = timeout
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def create_environment(self, task: RLTask) -> SandboxEnvironment:
        """Create a sandbox environment for a SWE-bench task."""
        repo_url = SWEBENCH_REPOS.get(task.repo, "")
        if not repo_url:
            logger.warning("Unknown repo %s, using direct URL", task.repo)
            repo_url = f"https://github.com/{task.repo}.git"

        env = SandboxEnvironment(
            use_docker=self.use_docker,
            timeout=self.timeout,
        )
        env.setup(
            repo_url=repo_url,
            base_commit=task.base_commit,
        )

        for cmd in task.setup_commands:
            env.run_command(cmd)

        return env

    def get_test_command(self, task: RLTask) -> str:
        """Construct the test command for verifying the task solution."""
        if task.fail_to_pass:
            test_files = " ".join(task.fail_to_pass)
            return f"python -m pytest {test_files} -x --timeout=60"
        return task.test_command
