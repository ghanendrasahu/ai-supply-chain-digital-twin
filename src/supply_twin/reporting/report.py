"""Report generation: consequence narrative, KPI comparison and charts.

Uses the non-interactive matplotlib ``Agg`` backend so charts render headlessly
in containers and CI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from supply_twin.config import Settings  # noqa: E402
from supply_twin.schemas import SimulationResult  # noqa: E402

_CMAP = {"baseline": "#1f77b4", "scenario": "#d62728"}


def consequence_summary(result: SimulationResult) -> dict[str, float | str]:
    chain = result.consequence_chain
    steps = {s.metric: s for s in chain.steps}

    def _impact(metric: str) -> float:
        step = steps.get(metric)
        return step.impact_value if step is not None else 0.0

    return {
        "intervention": chain.intervention,
        "chain": chain.headline(),
        "production_delta_pct": _impact("Production"),
        "inventory_delta_pct": _impact("Inventory"),
        "stockout_delta_pct": _impact("Stockout probability"),
        "revenue_impact": _impact("Revenue"),
        "horizon_days": result.horizon_days,
        "n_runs": result.n_runs,
        "run_id": result.run_id,
    }


def build_charts(result: SimulationResult, out_dir: Path) -> list[Path]:
    """Render inventory / revenue / production timelines and persist PNGs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    days = list(range(result.horizon_days))
    b, s = result.baseline_kpis, result.scenario_kpis

    series = ("inventory_total", "revenue", "production")
    titles = ("Total inventory", "Daily revenue", "Daily production")
    paths: list[Path] = []
    for metric, title in zip(series, titles, strict=False):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(days, [getattr(x, metric) for x in b], label="Baseline", color=_CMAP["baseline"])
        ax.plot(days, [getattr(x, metric) for x in s], label="Scenario", color=_CMAP["scenario"])
        ax.set_title(f"{title} — {result.intervention_summary}")
        ax.set_xlabel("Day")
        ax.legend()
        fig.tight_layout()
        path = out_dir / f"{result.run_id}_{metric}.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def render_markdown(
    result: SimulationResult,
    settings: Settings,
    kpis: dict[str, float] | None = None,
) -> str:
    """Render the report as a markdown string."""
    c = consequence_summary(result)
    lines = [
        "# AI Supply Chain Digital Twin — What-if Report",
        "",
        f"- **Run ID**: `{result.run_id}`",
        f"- **Intervention**: {c['intervention']}",
        f"- **Horizon**: {c['horizon_days']} days",
        f"- **Monte-Carlo runs**: {c['n_runs']}",
        "",
        "## Consequence chain",
        "",
        "```",
        c["chain"],
        "```",
        "",
        "| Metric | Change |",
        "| --- | --- |",
    ]
    for metric, value in (
        ("Production", c["production_delta_pct"]),
        ("Inventory", c["inventory_delta_pct"]),
        ("Stockout probability", c["stockout_delta_pct"]),
    ):
        value = float(value)
        sign = "+" if value >= 0 else ""
        lines.append(f"| {metric} | {sign}{value:.0f}% |")
    lines.append(f"| Revenue impact ({settings.currency}) | {_money(float(c['revenue_impact']))} |")
    lines.append("")

    lines.append("## Monte-Carlo uncertainty bands")
    lines.append("")
    lines.append("| Metric | p10 | p50 | p90 | mean |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for band in result.bands:
        if band.metric == "revenue_impact":
            fmt = lambda v: _money(v)  # noqa: E731
        else:
            fmt = lambda v: f"{v:+.1f}%"  # noqa: E731
        lines.append(
            f"| {band.metric} | {fmt(band.p10)} | {fmt(band.p50)} | {fmt(band.p90)} | {fmt(band.mean)} |"
        )
    lines.append("")

    if kpis is not None:
        lines.append("## Steady-state KPIs")
        lines.append("")
        lines.append("| KPI | Value |")
        lines.append("| --- | --- |")
        for key, value in kpis.items():
            lines.append(f"| {key} | {value:,.2f} |")
        lines.append("")

    lines.append("## Note")
    lines.append(
        "Results are produced by a discrete-event simulation with paired "
        "baseline/scenario runs (common random numbers). Percentage metrics are "
        "measured over the disruption window; revenue impact is cumulative over "
        "the full horizon."
    )
    return "\n".join(lines)


def build_report(
    result: SimulationResult,
    settings: Settings,
    out_dir: Path,
    kpis: dict[str, float] | None = None,
) -> tuple[Path, list[Path]]:
    """Write the markdown report and charts; return (report_path, chart_paths)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    md = render_markdown(result, settings, kpis)
    report_path = out_dir / f"report_{result.run_id}.md"
    report_path.write_text(md, encoding="utf-8")
    chart_paths = build_charts(result, out_dir)
    return report_path, chart_paths


def _money(value: float) -> str:
    symbol = "₹"
    crore = value / 1e7
    sign = "-" if value < 0 else "+"
    return f"{sign}{symbol}{abs(crore):.1f}Cr"
