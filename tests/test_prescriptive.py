from supply_twin.prescriptive.advisor import (
    ActionType,
    PrescriptiveAdvisor,
)
from supply_twin.schemas import SupplierFailure
from supply_twin.twin import SupplyChainTwin


def test_prescriptive_advisor_generation(twin: SupplyChainTwin):
    advisor = PrescriptiveAdvisor(twin)
    actions = advisor.generate_actions([SupplierFailure(supplier_id="SUP-A", duration_days=14)])
    assert len(actions) == 3
    action_types = {a.action_type for a in actions}
    assert ActionType.EXPEDITE_SECONDARY in action_types
    assert ActionType.SAFETY_BUFFER_EXPANSION in action_types


def test_prescriptive_advisor_evaluation(twin: SupplyChainTwin):
    advisor = PrescriptiveAdvisor(twin)
    recs = advisor.evaluate_mitigations(
        [SupplierFailure(supplier_id="SUP-A", duration_days=14)],
        horizon=30,
        n_runs=2,
    )
    assert len(recs) > 0
    top = recs[0]
    assert top.net_benefit is not None
    assert top.roi_multiple >= 0
    assert 0.0 <= top.feasibility_score <= 1.0
