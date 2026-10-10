from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from supply_twin.schemas import (
    DemandShock,
    Intervention,
    SupplierFailure,
)
from supply_twin.twin import SupplyChainTwin


class ActionType(StrEnum):
    EXPEDITE_SECONDARY = "expedite_secondary"
    SAFETY_BUFFER_EXPANSION = "safety_buffer_expansion"
    DEMAND_SHAPING_PRICE = "demand_shaping_price"
    ALTERNATIVE_ROUTING = "alternative_routing"


@dataclass
class MitigationAction:
    action_type: ActionType
    title: str
    description: str
    estimated_cost: float
    implementation_days: int
    interventions: list[Intervention]


@dataclass
class MitigationRecommendation:
    action: MitigationAction
    gross_recovered_revenue: float
    net_benefit: float
    roi_multiple: float
    stockout_reduction_pct: float
    feasibility_score: float


class PrescriptiveAdvisor:
    def __init__(self, twin: SupplyChainTwin) -> None:
        self.twin = twin

    def generate_actions(
        self,
        disruptions: list[Intervention],
    ) -> list[MitigationAction]:
        actions: list[MitigationAction] = []
        for d in disruptions:
            if isinstance(d, SupplierFailure):
                actions.append(
                    MitigationAction(
                        action_type=ActionType.EXPEDITE_SECONDARY,
                        title=f"Expedite Secondary Supplier for {d.supplier_id}",
                        description=(
                            f"Reroute raw procurement to alternate suppliers with 15% expediting premium during {d.duration_days}d downtime"
                        ),
                        estimated_cost=3_500_000.0,
                        implementation_days=2,
                        interventions=[
                            SupplierFailure(
                                supplier_id=d.supplier_id,
                                duration_days=d.duration_days,
                                severity=0.35,
                            )
                        ],
                    )
                )
                actions.append(
                    MitigationAction(
                        action_type=ActionType.SAFETY_BUFFER_EXPANSION,
                        title="Dynamic Buffer Reserve Allocation",
                        description=(
                            "Mobilize regional distribution buffer stock to compensate for factory throughput lag"
                        ),
                        estimated_cost=1_200_000.0,
                        implementation_days=1,
                        interventions=[
                            SupplierFailure(
                                supplier_id=d.supplier_id,
                                duration_days=d.duration_days,
                                severity=0.60,
                            )
                        ],
                    )
                )
                actions.append(
                    MitigationAction(
                        action_type=ActionType.DEMAND_SHAPING_PRICE,
                        title="Demand Shaping & Priority Allocation",
                        description=(
                            "Temporary selective rationing to protect highest-margin customer fulfillment"
                        ),
                        estimated_cost=450_000.0,
                        implementation_days=1,
                        interventions=[
                            SupplierFailure(
                                supplier_id=d.supplier_id,
                                duration_days=d.duration_days,
                                severity=0.75,
                            ),
                            DemandShock(magnitude=-0.08, duration_days=d.duration_days),
                        ],
                    )
                )
        return actions

    def evaluate_mitigations(
        self,
        disruptions: list[Intervention],
        horizon: int = 90,
        n_runs: int = 20,
    ) -> list[MitigationRecommendation]:
        disrupted_result = self.twin.what_if(
            interventions=disruptions,
            horizon=horizon,
            n_runs=n_runs,
        )
        unmitigated_loss = abs(disrupted_result.consequence_chain.revenue_impact or 0.0)
        unmitigated_stockout_delta = next(
            (
                s.impact_value
                for s in disrupted_result.consequence_chain.steps
                if s.metric == "Stockout probability"
            ),
            0.0,
        )

        candidate_actions = self.generate_actions(disruptions)
        recommendations: list[MitigationRecommendation] = []

        for action in candidate_actions:
            mitigated_result = self.twin.what_if(
                interventions=action.interventions,
                horizon=horizon,
                n_runs=n_runs,
            )
            mitigated_loss = abs(mitigated_result.consequence_chain.revenue_impact or 0.0)
            mitigated_stockout_delta = next(
                (
                    s.impact_value
                    for s in mitigated_result.consequence_chain.steps
                    if s.metric == "Stockout probability"
                ),
                0.0,
            )

            gross_saved = max(0.0, unmitigated_loss - mitigated_loss)
            net_benefit = gross_saved - action.estimated_cost
            roi = (gross_saved / action.estimated_cost) if action.estimated_cost > 0 else 1.0
            stockout_red = max(0.0, unmitigated_stockout_delta - mitigated_stockout_delta)
            feasibility = max(0.5, 1.0 - (action.implementation_days * 0.08))

            recommendations.append(
                MitigationRecommendation(
                    action=action,
                    gross_recovered_revenue=round(gross_saved, 2),
                    net_benefit=round(net_benefit, 2),
                    roi_multiple=round(roi, 2),
                    stockout_reduction_pct=round(stockout_red, 1),
                    feasibility_score=round(feasibility, 2),
                )
            )

        recommendations.sort(key=lambda r: r.net_benefit, reverse=True)
        return recommendations
