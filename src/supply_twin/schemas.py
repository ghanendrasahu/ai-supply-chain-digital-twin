"""Typed domain models for the supply-chain digital twin.

These pydantic models are the shared contract between the data generator, the
ML modules, the simulation engine, the reporting layer and the API.
"""

from __future__ import annotations

from datetime import date, timedelta
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class NodeType(StrEnum):
    """The six echelons of the modelled supply chain."""

    SUPPLIER = "supplier"
    FACTORY = "factory"
    WAREHOUSE = "warehouse"
    DISTRIBUTION = "distribution"
    RETAILER = "retailer"
    CUSTOMER = "customer"


# Ordered pipeline used to build the default chain and to propagate shocks.
CHAIN_ORDER: tuple[NodeType, ...] = (
    NodeType.SUPPLIER,
    NodeType.FACTORY,
    NodeType.WAREHOUSE,
    NodeType.DISTRIBUTION,
    NodeType.RETAILER,
    NodeType.CUSTOMER,
)


class SupplyNode(BaseModel):
    """A physical node in the supply chain."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    type: NodeType
    region: str = "IN"
    capacity_per_day: float = Field(gt=0.0)
    reliability: float = Field(default=0.98, gt=0.0, le=1.0)
    lead_time_days: int = Field(default=2, ge=0)
    # Cost of holding one unit for one day at this node.
    holding_cost_per_unit_day: float = Field(default=0.05, ge=0.0)


class SKU(BaseModel):
    """A stock-keeping unit."""

    model_config = ConfigDict(frozen=True)

    code: str
    name: str
    category: str = "general"
    unit_cost: float = Field(gt=0.0)
    price: float = Field(gt=0.0)
    # Own-price elasticity (negative). %Δqty per 1% Δprice.
    elasticity: float = Field(default=-1.5)

    @property
    def unit_margin(self) -> float:
        return self.price - self.unit_cost


class Lane(BaseModel):
    """A transportation lane between two nodes."""

    model_config = ConfigDict(frozen=True)

    origin: str
    destination: str
    transit_days: float = Field(gt=0.0)
    transit_std: float = Field(default=0.0, ge=0.0)
    cost_per_unit: float = Field(default=0.0, ge=0.0)


class InventoryState(BaseModel):
    """Inventory position for a SKU at a node."""

    node_id: str
    sku: str
    on_hand: float = Field(ge=0.0)
    in_transit: float = Field(default=0.0, ge=0.0)
    reorder_point: float = Field(default=0.0, ge=0.0)
    safety_stock: float = Field(default=0.0, ge=0.0)

    @property
    def position(self) -> float:
        """Inventory position = on-hand + in-transit."""
        return self.on_hand + self.in_transit


class DemandPoint(BaseModel):
    """A single demand observation / forecast point."""

    day: date
    quantity: float = Field(ge=0.0)


class DemandSeries(BaseModel):
    """A demand history or forecast for a SKU at a node."""

    node_id: str
    sku: str
    points: list[DemandPoint]

    @property
    def dates(self) -> list[date]:
        return [p.day for p in self.points]

    @property
    def values(self) -> list[float]:
        return [p.quantity for p in self.points]


# --------------------------------------------------------------------------
# Interventions (what-if events)
# --------------------------------------------------------------------------


class _InterventionBase(BaseModel):
    label: str | None = None
    start_day: int = Field(default=0, ge=0)


class SupplierFailure(_InterventionBase):
    kind: Literal["supplier_failure"] = "supplier_failure"
    supplier_id: str
    duration_days: int = Field(gt=0)
    # Fraction of the supplier's output that is lost [0..1].
    severity: float = Field(default=1.0, gt=0.0, le=1.0)


class FactoryOutage(_InterventionBase):
    kind: Literal["factory_outage"] = "factory_outage"
    factory_id: str
    duration_days: int = Field(gt=0)
    severity: float = Field(default=1.0, gt=0.0, le=1.0)


class DemandShock(_InterventionBase):
    kind: Literal["demand_shock"] = "demand_shock"
    # Positive = surge, negative = slump, expressed as a fraction.
    magnitude: float = Field(ge=-1.0, le=5.0)
    duration_days: int | None = None


class TransportDelay(_InterventionBase):
    kind: Literal["transport_delay"] = "transport_delay"
    lane_origin: str
    lane_destination: str
    added_days: float = Field(gt=0.0)


class PriceChange(_InterventionBase):
    kind: Literal["price_change"] = "price_change"
    sku: str
    change_pct: float = Field(ge=-1.0, le=5.0)


Intervention = Annotated[
    SupplierFailure | FactoryOutage | DemandShock | TransportDelay | PriceChange,
    Field(discriminator="kind"),
]


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


class KPISnapshot(BaseModel):
    """KPIs captured for a single day of a simulation."""

    day: int
    production: float = 0.0
    demand: float = 0.0
    fulfilled: float = 0.0
    lost_sales: float = 0.0
    inventory_total: float = 0.0
    stockout_events: int = 0
    stockout_probability: float = 0.0
    revenue: float = 0.0


class ConsequenceStep(BaseModel):
    """One link in the causal consequence chain."""

    metric: str
    delta_pct: float
    direction: Literal["up", "down", "flat"]
    impact_units: str
    impact_value: float
    narrative: str


class ConsequenceChain(BaseModel):
    """The ordered chain from an intervention to a financial impact."""

    intervention: str
    steps: list[ConsequenceStep]

    @property
    def revenue_impact(self) -> float | None:
        for step in reversed(self.steps):
            if step.metric.lower().startswith("revenue"):
                return step.impact_value
        return None

    def headline(self) -> str:
        """Render the chain as an arrow diagram."""
        parts = [self.intervention] + [s.narrative for s in self.steps]
        return " → ".join(parts)


class MonteCarloBand(BaseModel):
    metric: str
    p10: float
    p50: float
    p90: float
    mean: float


class SimulationResult(BaseModel):
    """The full output of a what-if simulation run."""

    run_id: str
    intervention_summary: str
    horizon_days: int
    n_runs: int
    baseline_kpis: list[KPISnapshot]
    scenario_kpis: list[KPISnapshot]
    consequence_chain: ConsequenceChain
    bands: list[MonteCarloBand] = Field(default_factory=list)

    def baseline_totals(self) -> KPISnapshot:
        return _sum_kpis(self.baseline_kpis, day=0)

    def scenario_totals(self) -> KPISnapshot:
        return _sum_kpis(self.scenario_kpis, day=0)


def _sum_kpis(snapshots: list[KPISnapshot], day: int) -> KPISnapshot:
    total = KPISnapshot(day=day)
    for s in snapshots:
        total.production += s.production
        total.demand += s.demand
        total.fulfilled += s.fulfilled
        total.lost_sales += s.lost_sales
        total.stockout_events += s.stockout_events
        total.revenue += s.revenue
        total.inventory_total = s.inventory_total  # last day level
    return total


def daterange(start: date, days: int) -> list[date]:
    """Return ``days`` consecutive dates beginning at ``start``."""
    return [start + timedelta(days=i) for i in range(days)]
