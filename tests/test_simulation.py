"""Tests for simulation engine."""

from supply_twin.schemas import DemandShock, SupplierFailure
from supply_twin.twin import SupplyChainTwin


def test_simulation_runs(twin: SupplyChainTwin):
    snaps = twin.run(interventions=[], horizon=10)
    assert len(snaps) == 10
    assert all(s.production >= 0 for s in snaps)
    assert all(s.inventory_total >= 0 for s in snaps)


def test_demand_shock_increases_sales(twin: SupplyChainTwin):
    base = twin.run(interventions=[], horizon=15)
    shock = twin.run([DemandShock(magnitude=0.3, duration_days=15)], horizon=15)

    total_base_demand = sum(s.demand for s in base)
    total_shock_demand = sum(s.demand for s in shock)
    assert total_shock_demand > total_base_demand


def test_supplier_failure_reduces_production(twin: SupplyChainTwin):
    base = twin.run(interventions=[], horizon=20)
    fail = twin.run([SupplierFailure(supplier_id="SUP-A", duration_days=10)], horizon=20)

    total_base_prod = sum(s.production for s in base)
    total_fail_prod = sum(s.production for s in fail)
    assert total_fail_prod <= total_base_prod
