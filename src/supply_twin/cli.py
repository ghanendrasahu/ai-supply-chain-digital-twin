"""`supply-twin` command-line interface."""

from __future__ import annotations

import json
import sys

import typer
from rich.console import Console
from rich.table import Table

from supply_twin.config import get_settings
from supply_twin.reporting.report import build_report as _br
from supply_twin.schemas import (
    DemandShock,
    FactoryOutage,
    PriceChange,
    SupplierFailure,
    TransportDelay,
)
from supply_twin.twin import SupplyChainTwin

_reconfig = getattr(sys.stdout, "reconfigure", None)
if callable(_reconfig):
    _reconfig(encoding="utf-8", errors="replace")

app = typer.Typer(help="AI Supply Chain Digital Twin CLI")
console = Console()


def _default_twin() -> SupplyChainTwin:
    settings = get_settings()
    settings.ensure_dirs()
    return SupplyChainTwin(settings=settings)


@app.command("demo")
def demo(
    days: int = typer.Option(14, help="Supplier failure duration in days"),
    runs: int = typer.Option(1000, help="Monte-Carlo runs"),
    horizon: int = typer.Option(90, help="Simulation horizon in days"),
) -> None:
    """Run the canonical supplier-failure scenario and print the chain."""
    twin = _default_twin()
    intervention = SupplierFailure(supplier_id="SUP-A", duration_days=days)
    result = twin.what_if(
        interventions=[intervention],
        horizon=horizon,
        n_runs=runs,
    )
    console.print("\n[bold cyan]Consequence chain[/bold cyan]")
    console.print(f"  {result.consequence_chain.headline()}")
    _print_result(result, twin)


@app.command("simulate")
def simulate(
    event: str = typer.Option(
        "supplier_failure",
        help="supplier_failure|factory_outage|demand_shock|transport_delay|price_change",
    ),
    supplier: str = typer.Option("SUP-A"),
    days: int = typer.Option(14),
    magnitude: float = typer.Option(0.2),
    sku: str = typer.Option("SKU-001"),
    runs: int = typer.Option(1000),
    horizon: int = typer.Option(90),
) -> None:
    """Run an arbitrary what-if intervention."""
    twin = _default_twin()
    interventions = _build_intervention(
        event, supplier=supplier, days=days, magnitude=magnitude, sku=sku
    )
    result = twin.what_if(interventions, horizon=horizon, n_runs=runs)
    _print_result(result, twin)


@app.command("forecast")
def forecast(
    retailer: str = typer.Option("RET-1"),
    sku: str = typer.Option("SKU-001"),
    horizon: int = typer.Option(30),
) -> None:
    """Forecast demand for a retailer/SKU."""
    twin = _default_twin()
    result = twin.forecast(retailer, sku, horizon)
    table = Table(title=f"Demand forecast {retailer} / {sku}")
    table.add_column("Day")
    table.add_column("Forecast")
    table.add_column("Lower")
    table.add_column("Upper")
    for d, f, lo, hi in zip(
        result.dates, result.forecast, result.lower, result.upper, strict=False
    ):
        table.add_row(str(d), f"{f:.1f}", f"{lo:.1f}", f"{hi:.1f}")
    console.print(table)


@app.command("optimize")
def optimize(
    sku: str = typer.Option("SKU-001"),
    node: str = typer.Option("WH-1"),
    service_level: float = typer.Option(0.95),
) -> None:
    """Compute the inventory policy for a SKU/node."""
    twin = _default_twin()
    policy = twin.optimize(sku, node, service_level)
    table = Table(title=f"Inventory policy {sku} @ {node}")
    for key, value in policy.__dict__.items():
        table.add_row(key, f"{value:,}" if isinstance(value, (int, float)) else str(value))
    console.print(table)


@app.command("kpis")
def kpis() -> None:
    """Show steady-state KPIs."""
    twin = _default_twin()
    table = Table(title="Steady-state KPIs")
    table.add_column("KPI")
    table.add_column("Value")
    for key, value in twin.kpis().items():
        table.add_row(key, f"{value:,.2f}")
    console.print(table)


@app.command("json")
def to_json(
    event: str = typer.Option("supplier_failure"),
    runs: int = typer.Option(1000),
    horizon: int = typer.Option(90),
) -> None:
    """Dump the what-if result as JSON to stdout."""
    twin = _default_twin()
    interventions = _build_intervention(event)
    result = twin.what_if(interventions, horizon=horizon, n_runs=runs)
    console.print(json.dumps(result.model_dump(), indent=2, default=str))


@app.command("serve")
def serve(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the FastAPI service (uvicorn)."""
    import uvicorn

    uvicorn.run("supply_twin.api.main:app", host=host, port=port, reload=False)


@app.command("ui")
def ui(port: int = 8501) -> None:
    """Launch the interactive Streamlit web dashboard."""
    import subprocess
    import sys
    from pathlib import Path

    app_path = Path(__file__).parent / "ui" / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)]
    )


def _build_intervention(
    event: str,
    supplier: str = "SUP-A",
    days: int = 14,
    magnitude: float = 0.2,
    sku: str = "SKU-001",
):
    event = event.lower()
    if event == "supplier_failure":
        return [SupplierFailure(supplier_id=supplier, duration_days=days)]
    if event == "factory_outage":
        return [FactoryOutage(factory_id="FAC-1", duration_days=days)]
    if event == "demand_shock":
        return [DemandShock(magnitude=magnitude, duration_days=days)]
    if event == "transport_delay":
        return [
            TransportDelay(lane_origin="DC-1", lane_destination="RET-1", added_days=float(days))
        ]
    if event == "price_change":
        return [PriceChange(sku=sku, change_pct=magnitude)]
    raise typer.BadParameter(f"Unknown event: {event}")


def _print_result(result, twin: SupplyChainTwin) -> None:
    total_b = result.baseline_totals()
    total_s = result.scenario_totals()

    table = Table(title="Impact summary")
    table.add_column("Metric")
    table.add_column("Baseline")
    table.add_column("Scenario")
    table.add_column("Delta")
    for metric in ("production", "demand", "fulfilled", "lost_sales", "revenue"):
        bv = getattr(total_b, metric)
        sv = getattr(total_s, metric)
        delta = sv - bv
        table.add_row(
            metric,
            f"{bv:,.0f}",
            f"{sv:,.0f}",
            f"{delta:+,.0f}",
        )
    table.add_row(
        "stockout_events",
        f"{total_b.stockout_events}",
        f"{total_s.stockout_events}",
        f"{total_s.stockout_events - total_b.stockout_events:+d}",
    )
    console.print(table)

    bands = Table(title="Monte-Carlo bands (p10 / p50 / p90)")
    bands.add_column("Metric")
    bands.add_column("p10")
    bands.add_column("p50")
    bands.add_column("p90")
    for b in result.bands:
        bands.add_row(
            b.metric,
            f"{b.p10:,.2f}",
            f"{b.p50:,.2f}",
            f"{b.p90:,.2f}",
        )
    console.print(bands)

    settings = get_settings()
    settings.ensure_dirs()
    report_path, _ = _br(result, settings, settings.report_dir, kpis=twin.kpis())
    console.print(f"\n[green]Report written:[/green] {report_path}")


if __name__ == "__main__":
    app()
