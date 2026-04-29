"""Dr. GRPO advantage estimation — use OpenRLHF ``--advantage_estimator dr_grpo``.

The upstream OpenRLHF implementation matches the Composer 2 report:
group baseline without per-group std / length standardization quirks.

See: https://github.com/OpenRLHF/OpenRLHF

This module documents the expected CLI wiring only.
"""

ADVANTAGE_ESTIMATOR_DR_GRPO = "dr_grpo"

__all__ = ["ADVANTAGE_ESTIMATOR_DR_GRPO"]
