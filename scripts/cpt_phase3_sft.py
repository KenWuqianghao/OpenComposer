#!/usr/bin/env python3
"""Stage 1d: CPT-SFT — thin wrapper around ``sft_tool_use.py`` (Qwen ChatML)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="CPT-SFT stage (wraps sft_tool_use.py)")
    parser.add_argument(
        "--config",
        default="configs/qwen3_moe_cpt_phase3_sft.yaml",
        help="SFT yaml for CPT-SFT (default: full pipeline phase 3)",
    )
    args, forwarded = parser.parse_known_args()
    root = Path(__file__).resolve().parent
    sft = root / "sft_tool_use.py"
    cmd = [sys.executable, str(sft), "--config", args.config, *forwarded]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
