from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd

from supply_twin.connectors.base import (
    BaseERPConnector,
    ERPInventoryRecord,
    ERPSalesRecord,
    ERPSupplierRecord,
)

SourceType = str | Path | io.StringIO | io.BytesIO


class CSVERPConnector(BaseERPConnector):
    def load_suppliers(self, source: SourceType) -> list[ERPSupplierRecord]:
        df = pd.read_csv(source)
        records: list[ERPSupplierRecord] = []
        for _, row in df.iterrows():
            records.append(
                ERPSupplierRecord(
                    supplier_id=str(row["supplier_id"]),
                    name=str(row["name"]),
                    capacity_per_day=float(row["capacity_per_day"]),
                    lead_time_days=int(row["lead_time_days"]),
                    reliability=float(row.get("reliability", 0.95)),
                )
            )
        return records

    def load_inventory(self, source: SourceType) -> list[ERPInventoryRecord]:
        df = pd.read_csv(source)
        records: list[ERPInventoryRecord] = []
        for _, row in df.iterrows():
            records.append(
                ERPInventoryRecord(
                    node_id=str(row["node_id"]),
                    sku_code=str(row["sku_code"]),
                    quantity_on_hand=float(row["quantity_on_hand"]),
                    reorder_point=float(row.get("reorder_point", 0.0)),
                )
            )
        return records

    def load_sales(self, source: SourceType) -> pd.DataFrame:
        df = pd.read_csv(source)
        expected = {"date", "node_id", "sku_code", "quantity"}
        if not expected.issubset(set(df.columns)):
            missing = expected - set(df.columns)
            raise ValueError(f"Sales CSV missing required columns: {missing}")
        return df


class JSONERPConnector(BaseERPConnector):
    def _read_data(self, source: SourceType) -> dict | list:
        if isinstance(source, (str, Path)) and Path(str(source)).exists():
            with open(source, encoding="utf-8") as f:
                return json.load(f)
        if hasattr(source, "read"):
            content = source.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8")
            return json.loads(content)
        if isinstance(source, str):
            return json.loads(source)
        raise ValueError(f"Unsupported source format: {type(source)}")

    def load_suppliers(self, source: SourceType) -> list[ERPSupplierRecord]:
        raw = self._read_data(source)
        items = raw if isinstance(raw, list) else raw.get("suppliers", [])
        return [ERPSupplierRecord(**item) for item in items]

    def load_inventory(self, source: SourceType) -> list[ERPInventoryRecord]:
        raw = self._read_data(source)
        items = raw if isinstance(raw, list) else raw.get("inventory", [])
        return [ERPInventoryRecord(**item) for item in items]

    def load_sales(self, source: SourceType) -> pd.DataFrame:
        raw = self._read_data(source)
        items = raw if isinstance(raw, list) else raw.get("sales", [])
        records = [ERPSalesRecord(**item).model_dump() for item in items]
        return pd.DataFrame(records)
