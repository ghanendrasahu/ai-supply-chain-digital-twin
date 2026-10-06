"""Tests for risk models."""

from supply_twin.risk.models import DeliveryDelayModel, SupplierFailureModel


def test_supplier_failure_recovery():
    model = SupplierFailureModel(daily_failure_prob=0.0, seed=42)
    model.force_failure("SUP-A", days=3)

    # Day 1: down
    res1 = model.step(["SUP-A"])
    assert not res1["SUP-A"]

    # Day 2: down
    res2 = model.step(["SUP-A"])
    assert not res2["SUP-A"]

    # Day 3: down
    res3 = model.step(["SUP-A"])
    assert not res3["SUP-A"]

    # Day 4: back up
    res4 = model.step(["SUP-A"])
    assert res4["SUP-A"]


def test_delivery_delay_lognormal():
    model = DeliveryDelayModel(seed=42)
    transit = model.sample_transit(mean_days=5.0, std_days=1.0)
    assert transit > 0
    congested = model.sample_transit(mean_days=5.0, std_days=1.0, congestion=0.5)
    assert congested > 0
