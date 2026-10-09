"""Shared test fixtures."""

from __future__ import annotations

import pytest

from supply_twin.config import Settings
from supply_twin.network import SupplyChainNetwork, build_default_network
from supply_twin.twin import SupplyChainTwin


@pytest.fixture
def settings() -> Settings:
    return Settings(
        env="test",
        random_seed=42,
        horizon_days=30,
        monte_carlo_runs=10,
        default_service_level=0.95,
    )


@pytest.fixture
def network() -> SupplyChainNetwork:
    return build_default_network()


@pytest.fixture
def twin(settings: Settings) -> SupplyChainTwin:
    return SupplyChainTwin(settings=settings, history_days=60)
