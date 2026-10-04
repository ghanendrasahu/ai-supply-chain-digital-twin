"""Tests for demand forecasting."""

from datetime import date

from supply_twin.data.generator import generate_history
from supply_twin.forecasting.demand import (
    GradientBoostForecaster,
    backtest,
    seasonal_naive_forecast,
)


def test_seasonal_naive_forecast():
    series = generate_history("RET-1", "SKU-001", date(2026, 1, 1), 28, seed=42)
    res = seasonal_naive_forecast(series, horizon=14)
    assert len(res.forecast) == 14
    assert len(res.lower) == 14
    assert len(res.upper) == 14
    assert (res.lower <= res.forecast).all()
    assert (res.forecast <= res.upper).all()


def test_gradient_boosting_forecast():
    series = generate_history("RET-1", "SKU-001", date(2026, 1, 1), 60, seed=42)
    forecaster = GradientBoostForecaster()
    forecaster.fit(series)
    res = forecaster.forecast(series, horizon=10)
    assert len(res.forecast) == 10
    assert (res.forecast >= 0).all()


def test_backtest_evaluates_metrics():
    series = generate_history("RET-1", "SKU-001", date(2026, 1, 1), 45, seed=42)
    res = backtest(series, horizon=7)
    assert "mape" in res.metrics
    assert "rmse" in res.metrics
    assert res.metrics["mape"] >= 0
