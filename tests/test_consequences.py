"""Tests for the consequence chain and what-if engine."""

from supply_twin.schemas import SupplierFailure
from supply_twin.twin import SupplyChainTwin


def test_canonical_consequence_chain(twin: SupplyChainTwin):
    result = twin.what_if(
        [SupplierFailure(supplier_id="SUP-A", duration_days=14)],
        horizon=90,
        n_runs=5,
    )
    chain = result.consequence_chain
    assert "SUP-A fails for 14 days" in chain.intervention

    steps = {s.metric: s for s in chain.steps}
    assert "Production" in steps
    assert "Inventory" in steps
    assert "Stockout probability" in steps
    assert "Revenue" in steps

    assert -20.0 <= steps["Production"].impact_value <= -15.0
    assert -35.0 <= steps["Inventory"].impact_value <= -25.0
    assert 35.0 <= steps["Stockout probability"].impact_value <= 45.0
    assert steps["Revenue"].impact_value < 0
    assert len(result.bands) > 0
