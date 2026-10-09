"""Tests for the FastAPI endpoints."""

from fastapi.testclient import TestClient

from supply_twin.api.main import app

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_network():
    res = client.get("/network")
    assert res.status_code == 200
    data = res.json()
    assert "nodes" in data
    assert "lanes" in data
    assert len(data["nodes"]) > 0


def test_skus():
    res = client.get("/skus")
    assert res.status_code == 200
    assert len(res.json()) >= 2


def test_demand_forecast():
    res = client.get("/demand/forecast?retailer=RET-1&sku=SKU-001&horizon=7")
    assert res.status_code == 200
    data = res.json()
    assert len(data["points"]) == 7


def test_inventory_optimize():
    res = client.get("/inventory/optimize?sku=SKU-001&node=WH-1")
    assert res.status_code == 200
    data = res.json()
    assert "safety_stock" in data
    assert "reorder_point" in data


def test_simulate_what_if():
    payload = [
        {
            "kind": "supplier_failure",
            "supplier_id": "SUP-A",
            "duration_days": 14,
        }
    ]
    res = client.post("/simulate/what-if?horizon=30&n_runs=2", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "consequence_chain" in data
    assert len(data["consequence_chain"]["steps"]) == 4
