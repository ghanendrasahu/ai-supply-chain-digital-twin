"""AI Supply Chain Digital Twin.

A simulation-first digital twin of an end-to-end supply chain with demand
forecasting, inventory optimisation and a what-if disruption simulator.
"""

from supply_twin.config import Settings, get_settings

__all__ = ["Settings", "get_settings", "__version__"]

__version__ = "0.1.0"
