"""Demand forecasting.

Two complementary approaches:

* :func:`seasonal_naive_forecast` — robust baseline that repeats the last
  observed week; used when history is short or as a sanity floor.
* :class:`GradientBoostForecaster` — gradient-boosted trees over engineered
  lag / rolling / calendar features for higher accuracy when enough history
  is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from supply_twin.schemas import DemandSeries


@dataclass
class ForecastResult:
    """Point forecast plus uncertainty and accuracy metrics."""

    node_id: str
    sku: str
    dates: list[date]
    forecast: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    model: str
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return float(np.sum(self.forecast))


def _mape(actual: np.ndarray, pred: np.ndarray) -> float:
    if len(actual) == 0:
        return 0.0
    mask = actual > 1e-6
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)


def seasonal_naive_forecast(
    series: DemandSeries,
    horizon: int,
    season_length: int = 7,
    z: float = 1.645,
) -> ForecastResult:
    """Repeat the most recent seasonal cycle ``horizon`` days forward."""
    values = np.asarray(series.values, dtype=float)
    if len(values) == 0:
        raise ValueError("Cannot forecast an empty series")

    pattern = values[-season_length:]
    reps = int(np.ceil(horizon / season_length))
    forecast = np.tile(pattern, reps)[:horizon]

    resid = (
        values[season_length:] - values[:-season_length]
        if len(values) > season_length
        else np.array([0.0])
    )
    sigma = float(np.std(resid)) if resid.size else float(np.std(values))
    lower = np.maximum(0.0, forecast - z * sigma)
    upper = forecast + z * sigma

    start = series.dates[-1] + timedelta(days=1)
    return ForecastResult(
        node_id=series.node_id,
        sku=series.sku,
        dates=[start + timedelta(days=i) for i in range(horizon)],
        forecast=forecast,
        lower=lower,
        upper=upper,
        model="seasonal_naive",
    )


def _make_features(values: np.ndarray, dates: list[date]) -> tuple[np.ndarray, np.ndarray]:
    """Build lag/rolling/calendar features and the aligned target."""
    n = len(values)
    rows: list[list[float]] = []
    targets: list[float] = []
    for i in range(7, n):
        d = dates[i]
        rows.append(
            [
                values[i - 1],
                values[i - 2],
                values[i - 7],
                float(np.mean(values[max(0, i - 7) : i])),
                float(np.mean(values[max(0, i - 14) : i])),
                float(d.weekday()),
                float(d.month),
            ]
        )
        targets.append(values[i])
    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)


class GradientBoostForecaster:
    """Gradient-boosted forecaster with recursive multi-step prediction."""

    def __init__(self, z: float = 1.645, random_state: int = 42) -> None:
        self.z = z
        self.model = HistGradientBoostingRegressor(
            max_depth=4,
            learning_rate=0.08,
            max_iter=400,
            l2_regularization=1.0,
            random_state=random_state,
        )
        self._sigma: float = 0.0
        self._dates: list[date] = []

    def fit(self, series: DemandSeries) -> GradientBoostForecaster:
        values = np.asarray(series.values, dtype=float)
        if len(values) < 21:
            raise ValueError("Need at least 21 observations for GBM forecasting")
        x, y = _make_features(values, series.dates)
        self.model.fit(x, y)
        resid = y - self.model.predict(x)
        self._sigma = float(np.std(resid))
        self._dates = list(series.dates)
        return self

    def forecast(self, series: DemandSeries, horizon: int) -> ForecastResult:
        values = list(np.asarray(series.values, dtype=float))
        dates = list(series.dates)
        preds: list[float] = []
        for _step in range(horizon):
            next_date = dates[-1] + timedelta(days=1)
            row = np.asarray(
                [
                    [
                        values[-1],
                        values[-2],
                        values[-7],
                        float(np.mean(values[-7:])),
                        float(np.mean(values[-14:])),
                        float(next_date.weekday()),
                        float(next_date.month),
                    ]
                ],
                dtype=float,
            )
            yhat = float(self.model.predict(row)[0])
            yhat = max(0.0, yhat)
            preds.append(yhat)
            values.append(yhat)
            dates.append(next_date)

        forecast = np.asarray(preds, dtype=float)
        lower = np.maximum(0.0, forecast - self.z * self._sigma)
        upper = forecast + self.z * self._sigma
        start = series.dates[-1] + timedelta(days=1)
        return ForecastResult(
            node_id=series.node_id,
            sku=series.sku,
            dates=[start + timedelta(days=i) for i in range(horizon)],
            forecast=forecast,
            lower=lower,
            upper=upper,
            model="gradient_boosting",
        )

    def score(self, series: DemandSeries) -> dict[str, float]:
        """In-sample accuracy metrics (MAPE, RMSE)."""
        values = np.asarray(series.values, dtype=float)
        x, y = _make_features(values, series.dates)
        pred = self.model.predict(x)
        rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
        return {"mape": _mape(y, pred), "rmse": rmse}


def backtest(series: DemandSeries, horizon: int = 14) -> ForecastResult:
    """Fit on all-but-last ``horizon`` points and score on the held-out tail."""
    vals = list(series.values)
    if len(vals) <= horizon + 7:
        return seasonal_naive_forecast(series, horizon)
    train = DemandSeries(
        node_id=series.node_id,
        sku=series.sku,
        points=series.points[: len(vals) - horizon],
    )
    actual = np.asarray(vals[-horizon:], dtype=float)
    try:
        model = GradientBoostForecaster().fit(train)
        result = model.forecast(train, horizon)
    except ValueError:
        result = seasonal_naive_forecast(train, horizon)
    result.metrics = {
        "mape": _mape(actual, result.forecast),
        "rmse": float(np.sqrt(np.mean((actual - result.forecast) ** 2))),
    }
    return result
