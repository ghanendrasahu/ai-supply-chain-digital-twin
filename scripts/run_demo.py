"""End-to-end demo: the canonical₹2.7Cr supplier-failure scenario."""

from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from supply_twin.config import get_settings  # noqa: E402
from supply_twin.reporting.report import build_report  # noqa: E402
from supply_twin.schemas import SupplierFailure  # noqa: E402
from supply_twin.twin import SupplyChainTwin  # noqa: E402

console = Console()


def main() -> int:
    settings = get_settings()
    settings.ensure_dirs()
    twin = SupplyChainTwin(settings=settings)

    console.rule("AI Supply Chain Digital Twin — Demo")
    console.print("[bold cyan]Supply chain:[/bold cyan] Supplier → Factory → "
                  "Warehouse → Distribution → Retailer → Customer")

    result = twin.what_if(
        interventions=[SupplierFailure(supplier_id="SUP-A", duration_days=14)],
        horizon=90,
        n_runs=settings.monte_carlo_runs,
    )

    console.print("\n[bold yellow]Consequence chain[/bold yellow]")
    console.print(f"  {result.consequence_chain.headline()}")
    console.print()
    for step in result.consequence_chain.steps:
        console.print(f"  • {step.metric}: {step.impact_value} {step.impact_units}")

    kpis = twin.kpis()
    console.print("\n[bold green]Steady-state KPIs[/bold green]")
    for key, value in kpis.items():
        console.print(f"  • {key}: {value:,.2f}")

    report_path, charts = build_report(result, settings, settings.report_dir, kpis=kpis)
    console.print(f"\n[green]Report:[/green] {report_path}")
    for chart in charts:
        console.print(f"[green]Chart:[/green] {chart}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())