"""The :class:`SupplyChainTwin` facade.

Wires the network, the synthetic demand history, the forecasting and inventory
optimisation modules and the simulation / consequence engines into one
drop-in object used by the API, the CLI and the demo script.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np

from supply_twin.config import Settings, get_settings
from supply_twin.data.generator import DemandGenerator
from supply_twin.forecasting.demand import ForecastResult, GradientBoostForecaster
from supply_twin.inventory.optimizer import InventoryOptimizer, InventoryPolicy
from supply_twin.network import SupplyChainNetwork, build_default_network
from supply_twin.schemas import (
    DemandSeries,
    Intervention,
    KPISnapshot,
    NodeType,
    SimulationResult,
)
from supply_twin.simulation.consequences import ConsequenceEngine
from supply_twin.simulation.engine import DemandTable, TwinSimulator, build_demand_tables

# Total expected daily customer demand (across SKUs/retailers) chosen to run
# the chain at ~ high utilisation so supply shocks actually bind.
_SPLIT = {
    ("RET-1", "SKU-001"): 2860.0,
    ("RET-2", "SKU-001"): 2340.0,
    ("RET-1", "SKU-002"): 2365.0,
    ("RET-2", "SKU-002"): 1935.0,
}


class SupplyChainTwin:
    """A ready-to-use supply chain digital twin."""

    def __init__(
        self,
        network: SupplyChainNetwork | None = None,
        settings: Settings | None = None,
        history_days: int = 240,
        demand_split: dict[tuple[str, str], float] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.network = network or build_default_network()
        self.history_days = history_days
        self.demand_split = demand_split or _SPLIT
        self.retailers = [n.id for n in self.network.nodes_of_type(NodeType.RETAILER)]

        # Canonical demand history (used for forecasts and the headline report).
        self.demand_history: dict[tuple[str, str], DemandSeries] = {}
        for key, base in self.demand_split.items():
            retailer, sku = key
            series = DemandGenerator(base_level=base, seed=self.settings.random_seed)
            self.demand_history[key] = series.generate(
                node_id=retailer,
                sku=sku,
                start=_start_date(self.settings.random_seed),
                days=self.history_days,
            )

    # -- simulator construction ------------------------------------------
    def _make_demand_table(self, seed: int, horizon: int) -> DemandTable:
        raw: dict[tuple[str, str], np.ndarray] = {}
        for key, base in self.demand_split.items():
            retailer, sku = key
            gen = DemandGenerator(base_level=base, seed=seed)
            series = gen.generate(
                node_id=retailer,
                sku=sku,
                start=_start_date(seed),
                days=horizon,
            )
            raw[key] = np.asarray(series.values, dtype=float)
        return build_demand_tables(self.network, raw, horizon)

    def make_simulator(self, seed: int | None = None, horizon: int | None = None):
        """Build a :class:`TwinSimulator` with an (optionally seeded) demand path."""
        seed = self.settings.random_seed if seed is None else seed
        horizon = horizon or self.settings.horizon_days
        table = self._make_demand_table(seed, horizon)
        return TwinSimulator(self.network, table, self.settings)

    # -- public operations ------------------------------------------------
    def run(
        self,
        interventions: Sequence[Intervention] | None = None,
        horizon: int | None = None,
        seed: int | None = None,
    ) -> list[KPISnapshot]:
        return self.make_simulator(seed, horizon).run(interventions, horizon, seed)

    def what_if(
        self,
        interventions: Sequence[Intervention],
        horizon: int | None = None,
        n_runs: int | None = None,
        seed: int | None = None,
        label: str | None = None,
    ) -> SimulationResult:
        engine = ConsequenceEngine(
            self.make_simulator,
            self.settings,
            n_retailers=len(self.retailers),
        )
        return engine.evaluate(
            interventions,
            horizon=horizon,
            n_runs=n_runs,
            seed=seed,
            label=label,
        )

    def forecast(self, retailer: str, sku: str, horizon: int = 30) -> ForecastResult:
        series = self.demand_history[(retailer, sku)]
        model = GradientBoostForecaster(random_state=self.settings.random_seed)
        return model.fit(series).forecast(series, horizon)

    def optimize(self, sku: str, node: str, service_level: float | None = None) -> InventoryPolicy:
        sku_obj = self.network.skus[sku]
        node_obj = self.network.node(node)
        upstream = next(iter(self.network.upstream(node)), None)
        lead_time = self.network.lane(upstream, node).transit_days if upstream else 1.0

        mean = 0.0
        var = 0.0
        if node_obj.type == NodeType.RETAILER:
            key = (node, sku)
            mean = float(np.mean([p.quantity for p in self.demand_history[key].points]))
            var = float(np.var([p.quantity for p in self.demand_history[key].points]))
        else:
            for retailer in self.retailers:
                key = (retailer, sku)
                vals = [p.quantity for p in self.demand_history[key].points]
                mean += float(np.mean(vals))
                var += float(np.var(vals))

        optimizer = InventoryOptimizer(
            service_level=service_level or self.settings.default_service_level
        )
        return optimizer.optimize(
            sku=sku,
            node_id=node,
            mean_daily_demand=mean,
            demand_std=var**0.5,
            lead_time_days=lead_time,
            unit_cost=sku_obj.unit_cost,
        )

    def kpis(self) -> dict[str, float]:
        """Current steady-state KPI estimates (unconstrained run)."""
        snapshots = self.run(interventions=[], seed=self.settings.random_seed)
        revenue = sum(s.revenue for s in snapshots)
        stockout_rate = sum(s.stockout_events for s in snapshots) / max(
            1, len(self.retailers) * len(snapshots)
        )
        service_level = 1.0 - stockout_rate
        return {
            "avg_daily_revenue": revenue / len(snapshots),
            "total_revenue_horizon": revenue,
            "stockout_rate": stockout_rate,
            "effective_service_level": service_level,
            "avg_daily_demand": float(np.mean([s.demand for s in snapshots])),
            "avg_daily_fulfilled": float(np.mean([s.fulfilled for s in snapshots])),
        }


def _start_date(seed: int) -> date:
    """A stable calendar start for a given seed (fixed epoch)."""
    del seed
    return date(2026, 1, 1)
