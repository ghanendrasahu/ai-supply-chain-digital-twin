"""Tests for the synthetic demand generator."""

from datetime import date

from supply_twin.data.generator import DemandGenerator, generate_history


def test_generator_deterministic():
    gen1 = DemandGenerator(base_level=500, seed=123)
    gen2 = DemandGenerator(base_level=500, seed=123)
    d1 = gen1.generate("RET-1", "SKU-001", date(2026, 1, 1), 30)
    d2 = gen2.generate("RET-1", "SKU-001", date(2026, 1, 1), 30)
    assert [p.quantity for p in d1.points] == [p.quantity for p in d2.points]


def test_generator_non_negative():
    series = generate_history("RET-1", "SKU-001", date(2026, 1, 1), 60, seed=42)
    assert len(series.points) == 60
    assert all(p.quantity >= 0 for p in series.points)
    assert series.node_id == "RET-1"
    assert series.sku == "SKU-001"
