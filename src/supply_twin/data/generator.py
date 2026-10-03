"""Synthetic demand / sales history generator.

Produces realistic-looking daily demand with:
  * a slow linear trend,
  * weekly seasonality (weekend lift),
  * annual seasonality (festive peak),
  * sparse promotional spikes,
  * multiplicative lognormal noise.

Everything is seeded so experiments are reproducible.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from supply_twin.schemas import DemandPoint, DemandSeries, daterange

# Monday .. Sunday seasonal multipliers.
_WEEKLY = np.array([1.00, 1.02, 1.03, 1.05, 1.12, 1.28, 1.20])


class DemandGenerator:
    """Generate a demand series for one SKU at one node."""

    def __init__(
        self,
        base_level: float = 500.0,
        trend_per_day: float = 0.05,
        noise_cv: float = 0.12,
        promo_prob: float = 0.03,
        promo_lift: float = 1.8,
        annual_amplitude: float = 0.18,
        seed: int | None = None,
    ) -> None:
        self.base_level = base_level
        self.trend_per_day = trend_per_day
        self.noise_cv = noise_cv
        self.promo_prob = promo_prob
        self.promo_lift = promo_lift
        self.annual_amplitude = annual_amplitude
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        node_id: str,
        sku: str,
        start: date,
        days: int,
    ) -> DemandSeries:
        """Return a fully-formed :class:`DemandSeries`."""
        dates = daterange(start, days)
        idx = np.arange(days)

        level = self.base_level + self.trend_per_day * idx
        weekly = np.array([_WEEKLY[d.weekday()] for d in dates])
        day_of_year = np.array([d.timetuple().tm_yday for d in dates])
        annual = 1.0 + self.annual_amplitude * np.sin(2 * np.pi * (day_of_year - 300) / 365.0)

        promo = np.ones(days)
        hits = self.rng.random(days) < self.promo_prob
        promo[hits] = self.promo_lift

        noise = self.rng.lognormal(mean=0.0, sigma=self.noise_cv, size=days)
        values = np.maximum(0.0, level * weekly * annual * promo * noise)

        points = [
            DemandPoint(day=d, quantity=round(float(v), 2))
            for d, v in zip(dates, values, strict=False)
        ]
        return DemandSeries(node_id=node_id, sku=sku, points=points)


def generate_history(
    node_id: str,
    sku: str,
    start: date,
    days: int,
    base_level: float = 500.0,
    seed: int | None = None,
) -> DemandSeries:
    """Convenience wrapper returning a fully-formed :class:`DemandSeries`."""
    gen = DemandGenerator(base_level=base_level, seed=seed)
    return gen.generate(node_id=node_id, sku=sku, start=start, days=days)
