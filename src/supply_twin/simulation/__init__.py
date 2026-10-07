"""Simulation package: discrete-event what-if engine and consequence chain."""

from supply_twin.simulation.consequences import ConsequenceEngine
from supply_twin.simulation.engine import TwinSimulator, build_demand_tables

__all__ = ["TwinSimulator", "ConsequenceEngine", "build_demand_tables"]
