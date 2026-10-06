"""Stochastic risk models.

* :class:`SupplierFailureModel` — two-state (up/down) availability process with
  a Markov-style per-day failure probability and Weibull recovery time.
* :class:`DeliveryDelayModel` — lognormal lane transit times with an optional
  multiplicative congestion factor.
"""

from __future__ import annotations

import numpy as np


def weibull_recovery_days(
    shape: float = 1.5,
    scale: float = 10.0,
    rng: np.random.Generator | None = None,
) -> float:
    """Draw a recovery time (days) from a Weibull distribution."""
    rng = rng or np.random.default_rng()
    return float(scale * rng.weibull(shape))


class SupplierFailureModel:
    """Model random, unplanned supplier downtime."""

    def __init__(
        self,
        daily_failure_prob: float = 0.01,
        recovery_shape: float = 1.5,
        recovery_scale: float = 10.0,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= daily_failure_prob <= 1.0:
            raise ValueError("daily_failure_prob must be in [0, 1]")
        self.daily_failure_prob = daily_failure_prob
        self.recovery_shape = recovery_shape
        self.recovery_scale = recovery_scale
        self.rng = np.random.default_rng(seed)
        # supplier_id -> remaining downtime days
        self._downtime: dict[str, float] = {}

    def step(self, supplier_ids: list[str]) -> dict[str, bool]:
        """Advance one day; return availability per supplier."""
        availability: dict[str, bool] = {}
        for sid in supplier_ids:
            remaining = self._downtime.get(sid, 0.0)
            if remaining > 0:
                self._downtime[sid] = remaining - 1
                availability[sid] = False
                continue
            if self.rng.random() < self.daily_failure_prob:
                down = weibull_recovery_days(self.recovery_shape, self.recovery_scale, self.rng)
                self._downtime[sid] = max(1.0, down) - 1
                availability[sid] = False
            else:
                availability[sid] = True
        return availability

    def force_failure(self, supplier_id: str, days: float) -> None:
        """Inject a deterministic outage (used by the what-if engine)."""
        self._downtime[supplier_id] = float(days)

    def time_to_failure(self, n_days: int) -> np.ndarray:
        """Simulate waiting times to first failure (for risk reporting)."""
        return self.rng.geometric(self.daily_failure_prob, size=n_days)


class DeliveryDelayModel:
    """Model lane transit time as a lognormal distribution."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def sample_transit(
        self,
        mean_days: float,
        std_days: float,
        congestion: float = 0.0,
    ) -> float:
        """Draw a transit time; ``congestion`` inflates the mean by a fraction."""
        if std_days <= 0:
            base = mean_days
        else:
            # Convert desired mean/std to lognormal mu/sigma.
            var = std_days**2
            mu = np.log(mean_days**2 / np.sqrt(var + mean_days**2))
            sigma = np.sqrt(np.log(1 + var / mean_days**2))
            base = float(self.rng.lognormal(mu, sigma))
        return max(0.0, base * (1.0 + congestion))
