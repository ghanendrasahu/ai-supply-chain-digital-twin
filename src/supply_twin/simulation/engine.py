"""Discrete-event, multi-echelon supply-chain simulator.

The engine advances a daily clock over a configurable horizon. Suppliers push
raw material into a factory; finished goods flow downstream through warehouse,
distribution and retailer echelons via transportation lanes with lead times.
Every downstream node replenishes inventory using a base-stock (order-up-to)
policy. Customer demand is fulfilled from retailer stock; unmet demand is lost.

Interventions (supplier failures, factory outages, demand shocks, transport
delays, price changes) are applied as time-indexed multipliers, so a single run
is fully deterministic given a seed and its inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from supply_twin.config import Settings, get_settings
from supply_twin.network import SupplyChainNetwork
from supply_twin.schemas import (
    DemandShock,
    FactoryOutage,
    Intervention,
    KPISnapshot,
    NodeType,
    PriceChange,
    SupplierFailure,
    TransportDelay,
)

STORAGE_TYPES = (NodeType.WAREHOUSE, NodeType.DISTRIBUTION, NodeType.RETAILER)


@dataclass
class DemandTable:
    """Per-(retailer, sku) daily demand arrays and summary statistics."""

    series: dict[tuple[str, str], np.ndarray]
    mean: dict[tuple[str, str], float] = field(default_factory=dict)
    std: dict[tuple[str, str], float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key, arr in self.series.items():
            self.mean[key] = float(np.mean(arr))
            self.std[key] = float(np.std(arr))


def build_demand_tables(
    network: SupplyChainNetwork,
    demand_series: dict[tuple[str, str], np.ndarray],
    horizon: int,
) -> DemandTable:
    """Slice / pad raw demand arrays to ``horizon`` days."""
    del network  # retained for signature symmetry / future expansion
    sliced: dict[tuple[str, str], np.ndarray] = {}
    for key, arr in demand_series.items():
        arr = np.asarray(arr, dtype=float)
        if len(arr) >= horizon:
            sliced[key] = arr[:horizon].copy()
        else:
            reps = int(np.ceil(horizon / max(1, len(arr))))
            sliced[key] = np.tile(arr, reps)[:horizon].copy()
    return DemandTable(series=sliced)


@dataclass
class _InterventionEffects:
    """Time-indexed multipliers derived from interventions."""

    supplier_availability: dict[str, np.ndarray] = field(default_factory=dict)
    factory_availability: dict[str, np.ndarray] = field(default_factory=dict)
    demand_multiplier: np.ndarray | None = None
    lane_delay: dict[tuple[str, str], float] = field(default_factory=dict)
    price_multiplier: dict[str, float] = field(default_factory=dict)


def _safe_stock(demand_std: float, lead_time: float, service_level: float) -> float:
    z = float(stats.norm.ppf(service_level))
    return z * demand_std * (max(lead_time, 1.0) ** 0.5)


class TwinSimulator:
    """Run the digital twin forward in time under a set of interventions."""

    REVIEW_PERIOD = 4

    def __init__(
        self,
        network: SupplyChainNetwork,
        demand: DemandTable,
        settings: Settings | None = None,
    ) -> None:
        self.network = network
        self.demand = demand
        self.settings = settings or get_settings()
        self.skus = list(network.skus.values())
        self.retailers = [n.id for n in network.nodes_of_type(NodeType.RETAILER)]
        self.suppliers = [n.id for n in network.nodes_of_type(NodeType.SUPPLIER)]
        self.factories = [n.id for n in network.nodes_of_type(NodeType.FACTORY)]
        self._sku_weight = self._compute_sku_weights()

    # -- setup helpers ----------------------------------------------------
    def _compute_sku_weights(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for sku in self.skus:
            totals[sku.code] = sum(self.demand.mean.get((r, sku.code), 0.0) for r in self.retailers)
        grand = sum(totals.values()) or 1.0
        return {code: val / grand for code, val in totals.items()}

    def _upstream(self, node_id: str) -> str | None:
        preds = self.network.upstream(node_id)
        return preds[0] if preds else None

    def _lane_transit(self, origin: str, destination: str, lane_delay: dict) -> float:
        lane = self.network.lane(origin, destination)
        return lane.transit_days + lane_delay.get((origin, destination), 0.0)

    # -- intervention effects --------------------------------------------
    def _build_effects(
        self, interventions: list[Intervention], horizon: int
    ) -> _InterventionEffects:
        eff = _InterventionEffects()
        eff.demand_multiplier = np.ones(horizon)
        for sid in self.suppliers:
            eff.supplier_availability[sid] = np.ones(horizon)
        for fid in self.factories:
            eff.factory_availability[fid] = np.ones(horizon)

        for iv in interventions:
            if isinstance(iv, SupplierFailure):
                arr = np.ones(horizon)
                s, e = iv.start_day, min(horizon, iv.start_day + iv.duration_days)
                arr[s:e] = 1.0 - iv.severity
                eff.supplier_availability[iv.supplier_id] = arr
            elif isinstance(iv, FactoryOutage):
                arr = np.ones(horizon)
                s, e = iv.start_day, min(horizon, iv.start_day + iv.duration_days)
                arr[s:e] = 1.0 - iv.severity
                eff.factory_availability[iv.factory_id] = arr
            elif isinstance(iv, DemandShock):
                s = iv.start_day
                e = horizon if iv.duration_days is None else min(horizon, s + iv.duration_days)
                eff.demand_multiplier[s:e] *= 1.0 + iv.magnitude
            elif isinstance(iv, TransportDelay):
                key = (iv.lane_origin, iv.lane_destination)
                eff.lane_delay[key] = eff.lane_delay.get(key, 0.0) + iv.added_days
            elif isinstance(iv, PriceChange):
                eff.price_multiplier[iv.sku] = 1.0 + iv.change_pct
        return eff

    # -- demand helpers ---------------------------------------------------
    def _node_capacity(self, node_id: str, sku_code: str) -> float:
        return self.network.node(node_id).capacity_per_day * self._sku_weight[sku_code]

    def _downstream_retailers(self, node_id: str) -> list[str]:
        out: list[str] = []
        stack = [node_id]
        seen: set[str] = set()
        while stack:
            cur = stack.pop()
            for nxt in self.network.downstream(cur):
                if nxt in seen:
                    continue
                seen.add(nxt)
                if self.network.node(nxt).type == NodeType.RETAILER:
                    out.append(nxt)
                stack.append(nxt)
        return out

    def _downstream_demand_stats(self, node_id: str, sku_code: str) -> tuple[float, float]:
        node = self.network.node(node_id)
        if node.type == NodeType.RETAILER:
            key = (node_id, sku_code)
            return self.demand.mean.get(key, 0.0), self.demand.std.get(key, 0.0)
        mean = 0.0
        var = 0.0
        for r in self._downstream_retailers(node_id):
            key = (r, sku_code)
            mean += self.demand.mean.get(key, 0.0)
            var += self.demand.std.get(key, 0.0) ** 2
        return mean, var**0.5

    def _flow_order(self) -> list[str]:
        order: list[str] = []
        for t in (NodeType.RETAILER, NodeType.DISTRIBUTION, NodeType.WAREHOUSE, NodeType.FACTORY):
            order.extend(n.id for n in self.network.nodes_of_type(t))
        return order

    # -- run --------------------------------------------------------------
    def run(
        self,
        interventions: list[Intervention] | None = None,
        horizon: int | None = None,
        seed: int | None = None,
    ) -> list[KPISnapshot]:
        """Advance the twin ``horizon`` days; return per-day KPI snapshots."""
        interventions = interventions or []
        horizon = horizon or self.settings.horizon_days
        _ = np.random.default_rng(self.settings.random_seed if seed is None else seed)
        eff = self._build_effects(interventions, horizon)

        demand: dict[tuple[str, str], np.ndarray] = {}
        for key, raw_arr in self.demand.series.items():
            if len(raw_arr) >= horizon:
                arr = raw_arr[:horizon].copy()
            else:
                reps = int(np.ceil(horizon / max(1, len(raw_arr))))
                arr = np.tile(raw_arr, reps)[:horizon].copy()

            arr = arr * eff.demand_multiplier[:horizon]
            if key[1] in eff.price_multiplier:
                pm = eff.price_multiplier[key[1]]
                elasticity = self.network.skus[key[1]].elasticity
                arr = arr * max(0.0, 1.0 + elasticity * (pm - 1.0))
            demand[key] = arr

        per_sku = [self._run_sku(sku.code, demand, eff, horizon) for sku in self.skus]
        return self._merge_sku_snapshots(per_sku, horizon)

    def _run_sku(
        self,
        sku_code: str,
        demand: dict[tuple[str, str], np.ndarray],
        eff: _InterventionEffects,
        horizon: int,
    ) -> list[KPISnapshot]:
        sku = self.network.skus[sku_code]

        on_hand: dict[str, float] = {}
        pipeline: dict[str, list[list[float]]] = {}
        target: dict[str, float] = {}

        storage_nodes = [
            n.id
            for n in self.network.nodes.values()
            if n.type in (NodeType.FACTORY,) + STORAGE_TYPES
        ]
        for nid in storage_nodes:
            mean, std = self._downstream_demand_stats(nid, sku_code)
            up = self._upstream(nid)
            lt = self._lane_transit(up, nid, eff.lane_delay) if up else 0.0
            ss = _safe_stock(std, lt, self.settings.default_service_level)
            target[nid] = max(mean * (lt + self.REVIEW_PERIOD) + ss, mean)
            on_hand[nid] = target[nid]
            pipeline[nid] = []

        # Raw-material buffer at each factory.
        raw_inv: dict[str, float] = {
            fac: self._node_capacity(fac, sku_code) * 1.0 for fac in self.factories
        }
        raw_pipeline: dict[str, list[list[float]]] = {fac: [] for fac in self.factories}

        snapshots: list[KPISnapshot] = []

        for t in range(horizon):
            # A. Deliver finished-goods arrivals.
            for nid in storage_nodes:
                pending = [p for p in pipeline[nid] if p[0] <= t]
                if pending:
                    on_hand[nid] += sum(q for _, q in pending)
                    pipeline[nid] = [p for p in pipeline[nid] if p[0] > t]

            # A2. Deliver raw-material arrivals to the factory buffer.
            for fac in self.factories:
                pending = [p for p in raw_pipeline[fac] if p[0] <= t]
                if pending:
                    raw_inv[fac] += sum(q for _, q in pending)
                    raw_pipeline[fac] = [p for p in raw_pipeline[fac] if p[0] > t]

            # B. Suppliers push raw material toward the factory.
            for fac in self.factories:
                for sup in self.suppliers:
                    out = self._node_capacity(sup, sku_code) * eff.supplier_availability[sup][t]
                    if out <= 0:
                        continue
                    transit = self._lane_transit(sup, fac, eff.lane_delay)
                    arrival = t + max(1, int(round(transit)))
                    raw_pipeline[fac].append([arrival, out])
                # keep buffer bounded (do not hoard indefinitely)
                cap_buffer = self._node_capacity(fac, sku_code) * 4.0
                raw_inv[fac] = min(raw_inv[fac], cap_buffer)

            # C. Factory production (capacity- and input-constrained).
            production = 0.0
            for fac in self.factories:
                cap = self._node_capacity(fac, sku_code) * eff.factory_availability[fac][t]
                made = min(cap, raw_inv[fac])
                raw_inv[fac] -= made
                on_hand[fac] += made
                production += made

            # D. Downstream->upstream order / ship pass.
            sales = 0.0
            demand_total = 0.0
            lost = 0.0
            stockouts = 0
            stockout_probs: list[float] = []
            for nid in self._flow_order():
                node = self.network.node(nid)
                up = self._upstream(nid)

                if node.type == NodeType.RETAILER:
                    d = float(demand.get((nid, sku_code), np.zeros(horizon))[t])
                    demand_total += d
                    start_stock = on_hand[nid]
                    sold = min(d, start_stock)
                    on_hand[nid] -= sold
                    sales += sold
                    lost += d - sold
                    if d - sold > 1e-9:
                        stockouts += 1
                    mean_d = self.demand.mean.get((nid, sku_code), 1.0)
                    std_d = max(self.demand.std.get((nid, sku_code), 1.0), 1e-6)
                    position = start_stock + sum(q for _, q in pipeline[nid])
                    prob = 1.0 - float(stats.norm.cdf((position - mean_d) / std_d))
                    stockout_probs.append(min(1.0, max(0.0, prob)))

                if up is None or up in self.suppliers:
                    continue
                position = on_hand[nid] + sum(q for _, q in pipeline[nid])
                order = max(0.0, target[nid] - position)
                shipped = min(order, on_hand[up])
                if shipped > 0:
                    on_hand[up] -= shipped
                    transit = self._lane_transit(up, nid, eff.lane_delay)
                    arrival = t + max(1, int(round(transit)))
                    pipeline[nid].append([arrival, shipped])

            inventory_total = sum(on_hand.values()) + sum(raw_inv.values())
            snapshots.append(
                KPISnapshot(
                    day=t,
                    production=round(production, 2),
                    demand=round(demand_total, 2),
                    fulfilled=round(sales, 2),
                    lost_sales=round(lost, 2),
                    inventory_total=round(inventory_total, 2),
                    stockout_events=stockouts,
                    stockout_probability=round(float(np.mean(stockout_probs)), 4)
                    if stockout_probs
                    else 0.0,
                    revenue=round(sales * sku.price, 2),
                )
            )

        return snapshots

    @staticmethod
    def _merge_sku_snapshots(per_sku: list[list[KPISnapshot]], horizon: int) -> list[KPISnapshot]:
        merged = [KPISnapshot(day=t) for t in range(horizon)]
        n_skus = len(per_sku)
        for series in per_sku:
            for t in range(horizon):
                s = series[t]
                m = merged[t]
                m.production += s.production
                m.demand += s.demand
                m.fulfilled += s.fulfilled
                m.lost_sales += s.lost_sales
                m.stockout_events += s.stockout_events
                m.stockout_probability += s.stockout_probability
                m.revenue += s.revenue
                m.inventory_total += s.inventory_total
        for t in range(horizon):
            merged[t].stockout_probability = round(
                merged[t].stockout_probability / max(1, n_skus), 4
            )
        return merged
