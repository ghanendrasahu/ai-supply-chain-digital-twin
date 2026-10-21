"""Central configuration for the digital twin.

All tunable knobs (including the calibration constants that produce the
canonical consequence chain) live here so they are auditable and overridable
via environment variables with the ``SUPPLY_TWIN_`` prefix.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings, loaded from environment / .env."""

    model_config = SettingsConfigDict(
        env_prefix="SUPPLY_TWIN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: str = "local"
    random_seed: int = 42
    horizon_days: int = Field(default=90, ge=1, le=730)
    monte_carlo_runs: int = Field(default=1000, ge=1, le=20000)
    default_service_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    currency: str = "INR"

    # Observability
    log_json: bool = False
    log_level: str = "INFO"

    data_dir: Path = Path("./data")
    report_dir: Path = Path("./data/reports")

    # --- Calibration constants for the consequence chain ------------------
    # Multi-echelon amplification of a production shortfall as it drains
    # downstream inventory ("bullwhip" decay factor).
    inventory_decay_multiplier: float = Field(default=1.72, gt=0.0)
    # Share of the customer base that is exposed to a given stockout.
    affected_customer_share: float = Field(default=0.15, gt=0.0, le=1.0)
    # Fraction of lost revenue that is permanently churned.
    churn_multiplier: float = Field(default=3.2, gt=0.0)
    # Baseline (no-disruption) revenue per day used by the revenue model.
    baseline_daily_revenue: float = Field(default=1_250_000.0, gt=0.0)

    def ensure_dirs(self) -> None:
        """Create data/report directories if they do not exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
