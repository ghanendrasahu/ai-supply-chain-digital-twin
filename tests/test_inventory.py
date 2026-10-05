"""Tests for inventory optimization."""

import pytest

from supply_twin.inventory.optimizer import (
    InventoryOptimizer,
    eoq,
    normal_loss,
    safety_stock,
    z_score,
)


def test_z_score_bounds():
    assert round(z_score(0.95), 2) == 1.64
    assert round(z_score(0.99), 2) == 2.33
    with pytest.raises(ValueError):
        z_score(0.4)
    with pytest.raises(ValueError):
        z_score(1.0)


def test_safety_stock_monotonicity():
    ss_low = safety_stock(demand_std=10, lead_time=4, service_level=0.90)
    ss_high = safety_stock(demand_std=10, lead_time=4, service_level=0.99)
    assert ss_high > ss_low


def test_eoq_positive():
    q = eoq(annual_demand=10000, order_cost=500, holding_cost_per_unit=5)
    assert q > 0
    assert round(q, 1) == 1414.2


def test_normal_loss():
    assert normal_loss(0.0) > 0.0
    assert normal_loss(3.0) < 0.01


def test_optimizer_flow():
    opt = InventoryOptimizer(service_level=0.95)
    policy = opt.optimize(
        sku="SKU-001",
        node_id="WH-1",
        mean_daily_demand=100.0,
        demand_std=15.0,
        lead_time_days=4.0,
        unit_cost=50.0,
    )
    assert policy.safety_stock > 0
    assert policy.reorder_point > policy.safety_stock
    assert policy.order_quantity > 0
    assert 0.90 <= policy.expected_fill_rate <= 1.0
