"""Shared dependencies: a cached twin and an in-memory results registry."""

from __future__ import annotations

from supply_twin.config import Settings, get_settings
from supply_twin.schemas import SimulationResult
from supply_twin.twin import SupplyChainTwin

_twin: SupplyChainTwin | None = None
_results: dict[str, SimulationResult] = {}


def get_settings_dep() -> Settings:
    return get_settings()


def get_twin() -> SupplyChainTwin:
    global _twin
    if _twin is None:
        settings = get_settings()
        settings.ensure_dirs()
        _twin = SupplyChainTwin(settings=settings)
    return _twin


def store_result(result: SimulationResult) -> SimulationResult:
    _results[result.run_id] = result
    return result


def get_result(run_id: str) -> SimulationResult | None:
    return _results.get(run_id)
