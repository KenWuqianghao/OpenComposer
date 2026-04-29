"""KL estimator k1 — use OpenRLHF ``--kl_estimator k1`` (default).

k1 loss term: :math:`-\\log r` with :math:`r = p_{\\theta}/p_{ref}` (Composer 2 §4.1).
"""

KL_ESTIMATOR_K1 = "k1"

__all__ = ["KL_ESTIMATOR_K1"]
