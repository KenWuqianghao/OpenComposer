"""Reward computation for RL training based on test execution."""

from __future__ import annotations

import logging
import re
from typing import Any

from opencomposer.environments.docker_env import SandboxEnvironment

logger = logging.getLogger(__name__)


class RewardComputer:
    """Computes rewards based on test execution results in a sandbox."""

    def __init__(
        self,
        pass_reward: float = 1.0,
        fail_reward: float = -0.5,
        partial_reward: bool = True,
        timeout_penalty: float = -0.3,
    ):
        self.pass_reward = pass_reward
        self.fail_reward = fail_reward
        self.partial_reward = partial_reward
        self.timeout_penalty = timeout_penalty

    def compute(
        self,
        env: SandboxEnvironment,
        test_command: str,
        fail_to_pass: list[str] | None = None,
    ) -> tuple[float, dict[str, Any]]:
        """Run tests and compute reward.

        Returns:
            (reward, info_dict) where info_dict contains details about test results.
        """
        output, exit_code = env.run_command(test_command)
        info = {
            "test_output": output[:2000],
            "exit_code": exit_code,
        }

        if "timed out" in output.lower():
            info["reason"] = "timeout"
            return self.timeout_penalty, info

        # Parse pytest-style output
        passed, failed, errors = self._parse_pytest_output(output)
        info["tests_passed"] = passed
        info["tests_failed"] = failed
        info["tests_errors"] = errors
        total = passed + failed + errors

        if exit_code == 0 and failed == 0 and errors == 0:
            info["reason"] = "all_passed"
            return self.pass_reward, info

        if total == 0:
            # Could not parse output or no tests ran
            if exit_code == 0:
                info["reason"] = "exit_0_no_tests_parsed"
                return self.pass_reward * 0.5, info
            info["reason"] = "no_tests_ran"
            return self.fail_reward, info

        if self.partial_reward and passed > 0:
            fraction = passed / total
            reward = self.fail_reward + (self.pass_reward - self.fail_reward) * fraction
            info["reason"] = f"partial_{passed}/{total}"
            return reward, info

        info["reason"] = "all_failed"
        return self.fail_reward, info

    def _parse_pytest_output(self, output: str) -> tuple[int, int, int]:
        """Parse pytest summary line to extract pass/fail/error counts."""
        passed = failed = errors = 0

        # Pattern: "X passed", "X failed", "X error"
        m_passed = re.search(r"(\d+)\s+passed", output)
        m_failed = re.search(r"(\d+)\s+failed", output)
        m_errors = re.search(r"(\d+)\s+error", output)

        if m_passed:
            passed = int(m_passed.group(1))
        if m_failed:
            failed = int(m_failed.group(1))
        if m_errors:
            errors = int(m_errors.group(1))

        return passed, failed, errors
