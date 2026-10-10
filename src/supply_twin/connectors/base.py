from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from supply_twin.network import SupplyChainNetwork
from supply_twin.schemas import SKU, Lane, NodeType, SupplyNode


class ERPSupplierRecord(BaseModel):
    supplier_id: str
    name: str
    capacity_per_day: float = Field(gt=0)
    lead_time_days: int = Field(ge=0)
    reliability: float = Field(default=0.95, gt=0, le=1)


class ERPInventoryRecord(BaseModel):
    node_id: str
    sku_code: str
    quantity_on_hand: float = Field(ge=0)
    reorder_point: float = Field(default=0, ge=0)


class ERPSalesRecord(BaseModel):
    date: str
    node_id: str
    sku_code: str
    quantity: float = Field(ge=0)


class BaseERPConnector(ABC):
    @abstractmethod
    def load_suppliers(self, source: Any) -> list[ERPSupplierRecord]:
        raise NotImplementedError

    @abstractmethod
    def load_inventory(self, source: Any) -> list[ERPInventoryRecord]:
        raise NotImplementedError

    @abstractmethod
    def load_sales(self, source: Any) -> pd.DataFrame:
        raise NotImplementedError

    def build_network_from_records(
        self,
        suppliers: list[ERPSupplierRecord],
        skus: list[SKU] | None = None,
    ) -> SupplyChainNetwork:
        nodes: list[SupplyNode] = []
        for s in suppliers:
            nodes.append(
                SupplyNode(
                    id=s.supplier_id,
                    name=s.name,
                    type=NodeType.SUPPLIER,
                    capacity_per_day=s.capacity_per_day,
                    lead_time_days=s.lead_time_days,
                    reliability=s.reliability,
                )
            )

        nodes.append(
            SupplyNode(
                id="FAC-1",
                name="Primary Plant",
                type=NodeType.FACTORY,
                capacity_per_day=sum(s.capacity_per_day for s in suppliers),
                lead_time_days=2,
            )
        )
        nodes.append(
            SupplyNode(
                id="WH-1",
                name="Central Warehouse",
                type=NodeType.WAREHOUSE,
                capacity_per_day=sum(s.capacity_per_day for s in suppliers) * 1.5,
                lead_time_days=1,
            )
        )
        nodes.append(
            SupplyNode(
                id="DC-1",
                name="Regional DC",
                type=NodeType.DISTRIBUTION,
                capacity_per_day=sum(s.capacity_per_day for s in suppliers) * 1.2,
                lead_time_days=2,
            )
        )
        nodes.append(
            SupplyNode(
                id="RET-1",
                name="Retail Store A",
                type=NodeType.RETAILER,
                capacity_per_day=sum(s.capacity_per_day for s in suppliers) * 0.6,
                lead_time_days=1,
            )
        )
        nodes.append(
            SupplyNode(
                id="RET-2",
                name="Retail Store B",
                type=NodeType.RETAILER,
                capacity_per_day=sum(s.capacity_per_day for s in suppliers) * 0.5,
                lead_time_days=1,
            )
        )
        nodes.append(
            SupplyNode(
                id="CUST-1",
                name="Market Customer Group",
                type=NodeType.CUSTOMER,
                capacity_per_day=100_000,
                lead_time_days=0,
            )
        )

        lanes: list[Lane] = []
        for s in suppliers:
            lanes.append(
                Lane(
                    origin=s.supplier_id,
                    destination="FAC-1",
                    transit_days=float(s.lead_time_days),
                    cost_per_unit=35.0,
                )
            )
        lanes.append(Lane(origin="FAC-1", destination="WH-1", transit_days=2.0, cost_per_unit=12.0))
        lanes.append(Lane(origin="WH-1", destination="DC-1", transit_days=2.0, cost_per_unit=8.0))
        lanes.append(Lane(origin="DC-1", destination="RET-1", transit_days=1.0, cost_per_unit=5.0))
        lanes.append(Lane(origin="DC-1", destination="RET-2", transit_days=2.0, cost_per_unit=6.0))
        lanes.append(
            Lane(origin="RET-1", destination="CUST-1", transit_days=1.0, cost_per_unit=2.0)
        )
        lanes.append(
            Lane(origin="RET-2", destination="CUST-1", transit_days=1.0, cost_per_unit=2.0)
        )

        default_skus = skus or [
            SKU(code="SKU-001", name="Standard Unit", unit_cost=180.0, price=420.0),
            SKU(code="SKU-002", name="Premium Unit", unit_cost=260.0, price=650.0),
        ]

        sup_map = {"FAC-1": [s.supplier_id for s in suppliers]}
        return SupplyChainNetwork(nodes, lanes, default_skus, sup_map)
