import io
import json

from supply_twin.connectors.erp import CSVERPConnector, JSONERPConnector


def test_csv_erp_connector():
    connector = CSVERPConnector()
    csv_data = io.StringIO(
        "supplier_id,name,capacity_per_day,lead_time_days,reliability\n"
        "SUP-1,Vendor One,3000.0,5,0.98\n"
        "SUP-2,Vendor Two,2500.0,8,0.92\n"
    )
    suppliers = connector.load_suppliers(csv_data)
    assert len(suppliers) == 2
    assert suppliers[0].supplier_id == "SUP-1"
    assert suppliers[0].capacity_per_day == 3000.0

    net = connector.build_network_from_records(suppliers)
    assert "SUP-1" in net.nodes
    assert "FAC-1" in net.nodes
    assert len(net.lanes) > 0


def test_json_erp_connector():
    connector = JSONERPConnector()
    payload = json.dumps(
        {
            "suppliers": [
                {
                    "supplier_id": "SUP-X",
                    "name": "Global Components",
                    "capacity_per_day": 5000.0,
                    "lead_time_days": 10,
                    "reliability": 0.99,
                }
            ]
        }
    )
    suppliers = connector.load_suppliers(payload)
    assert len(suppliers) == 1
    assert suppliers[0].supplier_id == "SUP-X"
