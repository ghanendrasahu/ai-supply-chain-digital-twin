"""Supply-chain topology: the connected graph of nodes, SKUs and lanes."""

from __future__ import annotations

from collections import defaultdict

import networkx as nx

from supply_twin.schemas import (
    CHAIN_ORDER,
    SKU,
    InventoryState,
    Lane,
    NodeType,
    SupplyNode,
)


class SupplyChainNetwork:
    """A directed graph of supply-chain nodes connected by transportation lanes.

    The default topology models a single product family flowing through:

        Supplier A/B/C -> Factory -> Warehouse -> Distribution -> Retailer -> Customer

    Suppliers carry a ``capacity_share`` so a failure of one supplier can be
    translated into a proportional loss of the factory's input.
    """

    def __init__(
        self,
        nodes: list[SupplyNode],
        lanes: list[Lane],
        skus: list[SKU],
        suppliers_by_factory: dict[str, list[str]] | None = None,
    ) -> None:
        self.nodes = {n.id: n for n in nodes}
        self.skus = {s.code: s for s in skus}
        self.lanes = lanes
        self.graph = nx.DiGraph()
        for node in nodes:
            self.graph.add_node(node.id, type=node.type.value)
        for lane in lanes:
            self.graph.add_edge(
                lane.origin,
                lane.destination,
                transit_days=lane.transit_days,
                transit_std=lane.transit_std,
                cost_per_unit=lane.cost_per_unit,
            )
        self.suppliers_by_factory = suppliers_by_factory or {}
        self._lane_index: dict[tuple[str, str], Lane] = {
            (lane.origin, lane.destination): lane for lane in lanes
        }

    # -- lookups ----------------------------------------------------------
    def node(self, node_id: str) -> SupplyNode:
        return self.nodes[node_id]

    def nodes_of_type(self, node_type: NodeType) -> list[SupplyNode]:
        return [n for n in self.nodes.values() if n.type == node_type]

    def lane(self, origin: str, destination: str) -> Lane:
        return self._lane_index[(origin, destination)]

    def downstream(self, node_id: str) -> list[str]:
        """Immediate downstream neighbours."""
        return list(self.graph.successors(node_id))

    def upstream(self, node_id: str) -> list[str]:
        """Immediate upstream neighbours."""
        return list(self.graph.predecessors(node_id))

    # -- derived analytics ------------------------------------------------
    def suppliers_of(self, factory_id: str) -> list[str]:
        """Return supplier ids feeding a factory (explicit map or inferred)."""
        if factory_id in self.suppliers_by_factory:
            return list(self.suppliers_by_factory[factory_id])
        return [n for n in self.graph.predecessors(factory_id)]

    def supplier_capacity_share(self, factory_id: str, supplier_id: str) -> float:
        """Share of a factory's capacity supplied by one supplier.

        Derived from each supplier's ``capacity_per_day`` relative to the sum
        of its peers. Used to translate a supplier outage into a factory
        production shortfall.
        """
        suppliers = self.suppliers_of(factory_id)
        total = sum(self.nodes[s].capacity_per_day for s in suppliers)
        if total <= 0:
            return 0.0
        return self.nodes[supplier_id].capacity_per_day / total

    def initial_inventory(
        self,
        days_of_cover: float = 20.0,
        daily_demand_per_sku: float = 500.0,
    ) -> dict[tuple[str, str], InventoryState]:
        """Seed every retailer/warehouse with a stock of cover.

        Keys are ``(node_id, sku_code)``.
        """
        states: dict[tuple[str, str], InventoryState] = {}
        for node in self.nodes.values():
            if node.type in (NodeType.CUSTOMER, NodeType.SUPPLIER):
                continue
            for sku in self.skus.values():
                on_hand = days_of_cover * daily_demand_per_sku * node.capacity_per_day / 1000.0
                states[(node.id, sku.code)] = InventoryState(
                    node_id=node.id,
                    sku=sku.code,
                    on_hand=round(on_hand, 2),
                )
        return states


def build_default_network() -> SupplyChainNetwork:
    """Construct the canonical demo supply chain."""
    nodes: list[SupplyNode] = []

    suppliers = [
        ("SUP-A", "Supplier A", 1800.0, 0.97, 7),
        ("SUP-B", "Supplier B", 4100.0, 0.95, 9),
        ("SUP-C", "Supplier C", 4100.0, 0.92, 11),
    ]
    for sid, name, cap, rel, lt in suppliers:
        nodes.append(
            SupplyNode(
                id=sid,
                name=name,
                type=NodeType.SUPPLIER,
                region="IN-W",
                capacity_per_day=cap,
                reliability=rel,
                lead_time_days=lt,
                holding_cost_per_unit_day=0.02,
            )
        )

    nodes.append(
        SupplyNode(
            id="FAC-1",
            name="Factory 1",
            type=NodeType.FACTORY,
            region="IN-W",
            capacity_per_day=8500.0,
            reliability=0.99,
            lead_time_days=3,
            holding_cost_per_unit_day=0.04,
        )
    )
    nodes.append(
        SupplyNode(
            id="WH-1",
            name="Warehouse 1",
            type=NodeType.WAREHOUSE,
            region="IN-W",
            capacity_per_day=12000.0,
            reliability=0.995,
            lead_time_days=1,
            holding_cost_per_unit_day=0.05,
        )
    )
    nodes.append(
        SupplyNode(
            id="DC-1",
            name="Distribution Center 1",
            type=NodeType.DISTRIBUTION,
            region="IN-N",
            capacity_per_day=11000.0,
            reliability=0.99,
            lead_time_days=2,
            holding_cost_per_unit_day=0.06,
        )
    )
    for rid, region in [("RET-1", "IN-N"), ("RET-2", "IN-S")]:
        nodes.append(
            SupplyNode(
                id=rid,
                name=f"Retailer {rid[-1]}",
                type=NodeType.RETAILER,
                region=region,
                capacity_per_day=6000.0,
                reliability=0.99,
                lead_time_days=1,
                holding_cost_per_unit_day=0.09,
            )
        )
    nodes.append(
        SupplyNode(
            id="CUST-1",
            name="End Customers",
            type=NodeType.CUSTOMER,
            region="IN",
            capacity_per_day=100000.0,
            reliability=1.0,
            lead_time_days=0,
            holding_cost_per_unit_day=0.0,
        )
    )

    lanes = [
        Lane(
            origin="SUP-A", destination="FAC-1", transit_days=7, transit_std=1.5, cost_per_unit=40
        ),
        Lane(
            origin="SUP-B", destination="FAC-1", transit_days=9, transit_std=2.0, cost_per_unit=36
        ),
        Lane(
            origin="SUP-C", destination="FAC-1", transit_days=11, transit_std=2.5, cost_per_unit=33
        ),
        Lane(origin="FAC-1", destination="WH-1", transit_days=3, transit_std=0.5, cost_per_unit=12),
        Lane(origin="WH-1", destination="DC-1", transit_days=2, transit_std=0.4, cost_per_unit=8),
        Lane(origin="DC-1", destination="RET-1", transit_days=2, transit_std=0.6, cost_per_unit=6),
        Lane(origin="DC-1", destination="RET-2", transit_days=3, transit_std=0.8, cost_per_unit=7),
        Lane(
            origin="RET-1", destination="CUST-1", transit_days=1, transit_std=0.2, cost_per_unit=3
        ),
        Lane(
            origin="RET-2", destination="CUST-1", transit_days=1, transit_std=0.2, cost_per_unit=3
        ),
    ]

    skus = [
        SKU(code="SKU-001", name="Widget Basic", category="widgets", unit_cost=180.0, price=420.0),
        SKU(code="SKU-002", name="Widget Pro", category="widgets", unit_cost=260.0, price=650.0),
    ]

    suppliers_by_factory = {"FAC-1": ["SUP-A", "SUP-B", "SUP-C"]}
    return SupplyChainNetwork(nodes, lanes, skus, suppliers_by_factory)


def chain_path(network: SupplyChainNetwork) -> list[list[str]]:
    """Return all simple source→sink paths through the network.

    Sources are suppliers, sinks are customers. Useful for narrating the
    propagation path of a disruption.
    """
    sources = [n.id for n in network.nodes_of_type(NodeType.SUPPLIER)]
    sinks = [n.id for n in network.nodes_of_type(NodeType.CUSTOMER)]
    paths: list[list[str]] = []
    for src in sources:
        for sink in sinks:
            for path in nx.all_simple_paths(network.graph, src, sink):
                paths.append(path)
    return paths


def echelon_summary(network: SupplyChainNetwork) -> dict[str, int]:
    """Count of nodes per echelon, in pipeline order."""
    counts: dict[str, int] = defaultdict(int)
    for node in network.nodes.values():
        counts[node.type.value] += 1
    return {t.value: counts.get(t.value, 0) for t in CHAIN_ORDER}
