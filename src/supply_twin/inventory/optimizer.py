"""Inventory optimisation: safety stock, reorder point, EOQ, fill rate.

Classical continuous-review (Q,R) policy built on the normal loss function.
All formulas are exposed as pure functions for testability and composed by
:class:`InventoryOptimizer`.
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy import stats


def z_score(service_level: float) -> float:
    """Standard-normal quantile for a target cycle service level."""
    if not 0.5 < service_level < 1.0:
        raise ValueError("service_level must be in (0.5, 1.0)")
    return float(stats.norm.ppf(service_level))


def safety_stock(demand_std: float, lead_time: float, service_level: float) -> float:
    """Safety stock = z * sigma_demand * sqrt(lead_time)."""
    if demand_std < 0 or lead_time < 0:
        raise ValueError("demand_std and lead_time must be non-negative")
    return z_score(service_level) * demand_std * (lead_time**0.5)


def reorder_point(mean_demand: float, lead_time: float, ss: float) -> float:
    """ROP = expected demand over lead time + safety stock."""
    return mean_demand * lead_time + ss


def eoq(annual_demand: float, order_cost: float, holding_cost_per_unit: float) -> float:
    """Economic order quantity = sqrt(2*D*S / H)."""
    if annual_demand <= 0 or order_cost <= 0 or holding_cost_per_unit <= 0:
        raise ValueError("annual_demand, order_cost and holding_cost must be positive")
    return float((2 * annual_demand * order_cost / holding_cost_per_unit) ** 0.5)


def normal_loss(z: float) -> float:
    """Standard normal loss function L(z) = phi(z) - z * (1 - Phi(z))."""
    return float(stats.norm.pdf(z) - z * (1 - stats.norm.cdf(z)))


def expected_shortage_per_cycle(demand_std: float, lead_time: float, ss: float) -> float:
    """Expected units short per replenishment cycle for a (Q,R) policy."""
    sigma_lt = demand_std * (lead_time**0.5)
    if sigma_lt == 0:
        return 0.0
    z = ss / sigma_lt
    return float(sigma_lt * normal_loss(z))


@dataclass
class InventoryPolicy:
    """Recommended inventory policy for a SKU at a node."""

    sku: str
    node_id: str
    mean_daily_demand: float
    demand_std: float
    lead_time_days: float
    service_level: float
    safety_stock: float
    reorder_point: float
    order_quantity: float
    expected_fill_rate: float
    expected_shortage_per_cycle: float


class InventoryOptimizer:
    """Compose the classical policies into a single recommendation."""

    def __init__(
        self,
        service_level: float = 0.95,
        order_cost: float = 5000.0,
        holding_cost_rate: float = 0.20,
    ) -> None:
        self.service_level = service_level
        self.order_cost = order_cost
        self.holding_cost_rate = holding_cost_rate

    def optimize(
        self,
        sku: str,
        node_id: str,
        mean_daily_demand: float,
        demand_std: float,
        lead_time_days: float,
        unit_cost: float,
    ) -> InventoryPolicy:
        ss = safety_stock(demand_std, lead_time_days, self.service_level)
        rop = reorder_point(mean_daily_demand, lead_time_days, ss)

        holding_cost_per_unit = max(1e-6, unit_cost * self.holding_cost_rate)
        annual_demand = mean_daily_demand * 365.0
        q = eoq(annual_demand, self.order_cost, holding_cost_per_unit)

        short = expected_shortage_per_cycle(demand_std, lead_time_days, ss)
        fill_rate = 1.0 - (short / q) if q > 0 else 1.0

        return InventoryPolicy(
            sku=sku,
            node_id=node_id,
            mean_daily_demand=mean_daily_demand,
            demand_std=demand_std,
            lead_time_days=lead_time_days,
            service_level=self.service_level,
            safety_stock=round(ss, 2),
            reorder_point=round(rop, 2),
            order_quantity=round(q, 2),
            expected_fill_rate=round(max(0.0, min(1.0, fill_rate)), 4),
            expected_shortage_per_cycle=round(short, 3),
        )
