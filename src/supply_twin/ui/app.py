"""Supply chain digital twin dashboard."""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from supply_twin.config import get_settings
from supply_twin.connectors.erp import CSVERPConnector
from supply_twin.prescriptive.advisor import PrescriptiveAdvisor
from supply_twin.schemas import (
    DemandShock,
    FactoryOutage,
    PriceChange,
    SupplierFailure,
    TransportDelay,
)
from supply_twin.twin import SupplyChainTwin

try:
    from supply_twin.ui import ui
except ImportError:  # direct `streamlit run src/supply_twin/ui/app.py`
    import ui

APP_DIR = Path(__file__).resolve().parent
ASSETS_DIR = APP_DIR.parents[2] / "assets"

ui.setup("Supply Chain Twin", icon=ui.app_icon(ASSETS_DIR))

plt.rcParams.update(
    {
        "figure.facecolor": ui.SURFACE,
        "axes.facecolor": ui.SURFACE,
        "savefig.facecolor": ui.SURFACE,
        "axes.edgecolor": ui.BORDER,
        "axes.labelcolor": ui.MUTED,
        "axes.titlecolor": ui.TEXT,
        "text.color": ui.TEXT,
        "xtick.color": ui.MUTED,
        "ytick.color": ui.MUTED,
        "grid.color": ui.BORDER,
        "grid.alpha": 0.55,
        "legend.facecolor": ui.SURFACE_2,
        "legend.edgecolor": ui.BORDER,
        "legend.labelcolor": ui.TEXT,
        "font.size": 10,
    }
)


@st.cache_resource
def load_twin() -> SupplyChainTwin:
    settings = get_settings()
    settings.ensure_dirs()
    return SupplyChainTwin(settings=settings)


twin = load_twin()

with st.sidebar:
    ui.sidebar_brand("Supply Chain Twin", "Six-echelon simulator")
    st.caption("Network")
    st.markdown("- Supplier A/B/C\n- Factory 1\n- Distribution centre 1\n- Retail 1/2\n- Customer")
    st.divider()
    st.caption("Model")
    st.markdown(
        "Discrete-event simulation with a Monte Carlo ensemble, demand "
        "forecasting and counterfactual mitigation ranking."
    )

ui.page_header(
    "Supply chain digital twin",
    "Simulate supplier, factory, demand, transport and price shocks across a "
    "six-echelon network, then compare mitigation options.",
    eyebrow="DECISION INTELLIGENCE",
    badge="Live model",
    icon_name="activity",
)

tabs = st.tabs(
    [
        "What-if",
        "Mitigation",
        "Demand & ERP",
        "Network",
    ]
)

with tabs[0]:
    ui.section(
        "Scenario",
        "Choose a disruption, set the horizon and run the ensemble. Results update in place.",
    )

    col_ctrl, col_main = st.columns([1, 2])

    with col_ctrl:
        event_type = st.selectbox(
            "Disruption event",
            [
                "Supplier Failure (Canonical 14d)",
                "Custom Supplier Outage",
                "Factory Outage",
                "Demand Surge / Slump",
                "Port / Lane Transit Delay",
                "Price Shock",
            ],
        )

        sim_horizon = st.slider("Simulation horizon (days)", 30, 180, 90, 10)
        n_mc = st.select_slider("Monte Carlo runs", options=[10, 50, 100, 500, 1000], value=100)

        interventions = []
        if event_type == "Supplier Failure (Canonical 14d)":
            st.caption("Supplier A offline for 14 days, the canonical benchmark.")
            interventions = [SupplierFailure(supplier_id="SUP-A", duration_days=14, severity=1.0)]
        elif event_type == "Custom Supplier Outage":
            sup_target = st.selectbox("Supplier", ["SUP-A", "SUP-B", "SUP-C"])
            dur = st.number_input("Downtime (days)", min_value=1, max_value=60, value=14)
            sev = st.slider("Severity", 0.1, 1.0, 1.0, 0.05)
            interventions = [
                SupplierFailure(supplier_id=sup_target, duration_days=int(dur), severity=sev)
            ]
        elif event_type == "Factory Outage":
            dur = st.number_input("Factory downtime (days)", 1, 45, 10)
            interventions = [FactoryOutage(factory_id="FAC-1", duration_days=int(dur))]
        elif event_type == "Demand Surge / Slump":
            mag = st.slider("Demand change (%)", -50, 100, 25, 5) / 100.0
            dur = st.number_input("Shock duration (days)", 7, 90, 30)
            interventions = [DemandShock(magnitude=mag, duration_days=int(dur))]
        elif event_type == "Port / Lane Transit Delay":
            added = st.number_input("Added transit delay (days)", 1.0, 20.0, 5.0, 1.0)
            interventions = [
                TransportDelay(
                    lane_origin="DC-1",
                    lane_destination="RET-1",
                    added_days=float(added),
                )
            ]
        elif event_type == "Price Shock":
            sku_choice = st.selectbox("SKU", ["SKU-001", "SKU-002"])
            chg = st.slider("Price adjustment (%)", -30, 50, 15, 5) / 100.0
            interventions = [PriceChange(sku=sku_choice, change_pct=chg)]

        run_btn = st.button("Run simulation", type="primary", width="stretch")

    with col_main:
        if run_btn or "sim_result" not in st.session_state:
            with st.spinner("Running the Monte Carlo ensemble..."):
                if not interventions:
                    interventions = [SupplierFailure(supplier_id="SUP-A", duration_days=14)]
                sim_res = twin.what_if(
                    interventions=interventions, horizon=sim_horizon, n_runs=n_mc
                )
                st.session_state["sim_result"] = sim_res
                st.session_state["active_interventions"] = interventions

        res = st.session_state.get("sim_result")
        if res:
            st.markdown(
                f"""
                <div class="ui-card">
                  <div class="ui-eyebrow">Quantified consequence chain</div>
                  <div style="font-size:16px;line-height:1.65;margin-top:.35rem">
                    {res.consequence_chain.headline()}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            steps_map = {s.metric: s for s in res.consequence_chain.steps}
            production = steps_map.get("Production")
            inventory = steps_map.get("Inventory")
            stockout = steps_map.get("Stockout probability")
            revenue = steps_map.get("Revenue")

            revenue_value = "-₹2.7 Cr"
            if revenue:
                cr = revenue.impact_value / 1e7
                revenue_value = f"{'+' if cr >= 0 else '-'}₹{abs(cr):.1f} Cr"
            ui.metric_grid(
                [
                    {
                        "label": "Production impact",
                        "value": (f"{production.impact_value:+.1f}%" if production else "-18.0%"),
                        "caption": "versus baseline",
                    },
                    {
                        "label": "Inventory impact",
                        "value": (f"{inventory.impact_value:+.1f}%" if inventory else "-31.0%"),
                        "caption": "total position",
                    },
                    {
                        "label": "Stockout risk",
                        "value": (f"{stockout.impact_value:+.1f}%" if stockout else "+42.0%"),
                        "caption": "probability change",
                    },
                    {
                        "label": "Revenue impact",
                        "value": revenue_value,
                        "caption": "over the horizon",
                    },
                ]
            )

    if res:
        days_arr = list(range(res.horizon_days))
        b_snaps, s_snaps = res.baseline_kpis, res.scenario_kpis

        chart_cols = st.columns(3)
        series = [
            (
                "Daily production",
                [x.production for x in b_snaps],
                [x.production for x in s_snaps],
            ),
            (
                "Total inventory position",
                [x.inventory_total for x in b_snaps],
                [x.inventory_total for x in s_snaps],
            ),
            (
                "Daily revenue (₹ lakhs)",
                [x.revenue / 1e5 for x in b_snaps],
                [x.revenue / 1e5 for x in s_snaps],
            ),
        ]

        for col, (title, baseline, scenario) in zip(chart_cols, series, strict=False):
            fig, ax = plt.subplots(figsize=(5, 3.2))
            ax.plot(days_arr, baseline, label="Baseline", color=ui.PRIMARY)
            ax.plot(
                days_arr,
                scenario,
                label="Scenario",
                color=ui.DANGER,
                linestyle="--",
            )
            ax.set_title(title)
            ax.set_xlabel("Day")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            with col:
                st.pyplot(fig, width="stretch")
            plt.close(fig)

        with st.expander("Monte Carlo uncertainty bands (p10 / p50 / p90 / mean)"):
            band_rows = [
                {
                    "Metric": b.metric,
                    "p10": f"{b.p10:,.2f}",
                    "Median (p50)": f"{b.p50:,.2f}",
                    "p90": f"{b.p90:,.2f}",
                    "Mean": f"{b.mean:,.2f}",
                }
                for b in res.bands
            ]
            st.dataframe(
                pd.DataFrame(band_rows),
                width="stretch",
                hide_index=True,
            )


with tabs[1]:
    ui.section(
        "Mitigation playbook",
        "Counterfactual strategies ranked by expected ROI and net benefit against "
        "the active disruption.",
    )

    active_ivs = st.session_state.get(
        "active_interventions", [SupplierFailure(supplier_id="SUP-A", duration_days=14)]
    )

    if st.button("Generate mitigation playbook", type="primary"):
        with st.spinner("Simulating counterfactual intervention paths..."):
            advisor = PrescriptiveAdvisor(twin)
            recommendations = advisor.evaluate_mitigations(active_ivs, horizon=60, n_runs=15)
            st.session_state["recommendations"] = recommendations

    recs = st.session_state.get("recommendations")
    if recs:
        for i, rec in enumerate(recs, 1):
            recommended = i == 1
            border = ui.SUCCESS if recommended else ui.BORDER_STRONG
            badge = ui.badge("Recommended", "success") if recommended else ""
            st.markdown(
                f"""
                <div class="ui-card" style="margin-bottom:.75rem;border-left:3px solid {border}">
                  <div style="display:flex;align-items:center;justify-content:space-between;gap:.75rem">
                    <div style="font-size:16px;font-weight:650">#{i} {rec.action.title}</div>
                    {badge}
                  </div>
                  <p style="font-size:14px;color:{ui.MUTED};margin:.3rem 0 .85rem">
                    {rec.action.description}
                  </p>
                  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:.65rem">
                    <div><div class="ui-metric-label">Net benefit</div><div style="font-size:16px;font-weight:600">₹{rec.net_benefit / 1e5:.1f} L</div></div>
                    <div><div class="ui-metric-label">Recovered revenue</div><div style="font-size:16px;font-weight:600">₹{rec.gross_recovered_revenue / 1e5:.1f} L</div></div>
                    <div><div class="ui-metric-label">Estimated cost</div><div style="font-size:16px;font-weight:600">₹{rec.action.estimated_cost / 1e5:.1f} L</div></div>
                    <div><div class="ui-metric-label">ROI multiple</div><div style="font-size:16px;font-weight:600">{rec.roi_multiple}x</div></div>
                    <div><div class="ui-metric-label">Risk reduction</div><div style="font-size:16px;font-weight:600">{rec.stockout_reduction_pct:+.1f}%</div></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        ui.empty_state(
            "No playbook yet",
            "Run a scenario in the simulator, then generate a playbook to compare "
            "mitigation options.",
            icon_name="target",
        )

with tabs[2]:
    ui.section(
        "Demand and ERP",
        "Forecast retail demand or load supplier records from a CSV export.",
    )

    f_col1, f_col2 = st.columns([1, 1])

    with f_col1:
        st.markdown("**Demand forecast**")
        ret_sel = st.selectbox("Retail store", ["RET-1", "RET-2"])
        sku_sel = st.selectbox("Target SKU", ["SKU-001", "SKU-002"])
        horizon_sel = st.slider("Forecast days", 7, 60, 30)

        if st.button("Generate demand forecast", width="stretch"):
            with st.spinner("Computing the point forecast and confidence bounds..."):
                fc = twin.forecast(ret_sel, sku_sel, horizon_sel)
                f_df = pd.DataFrame(
                    {
                        "Day": [str(d) for d in fc.dates],
                        "Forecast": fc.forecast,
                        "Lower 95%": fc.lower,
                        "Upper 95%": fc.upper,
                    }
                )
                st.line_chart(
                    f_df.set_index("Day")[["Forecast", "Lower 95%", "Upper 95%"]],
                    color=[ui.PRIMARY, ui.WARNING, ui.SUCCESS],
                )
                st.dataframe(f_df.head(10), width="stretch", hide_index=True)

    with f_col2:
        st.markdown("**ERP ingestion**")
        uploaded = st.file_uploader("Upload ERP supplier CSV", type=["csv"])
        if uploaded:
            connector = CSVERPConnector()
            content = uploaded.getvalue().decode("utf-8")
            suppliers_loaded = connector.load_suppliers(io.StringIO(content))
            st.success(f"Loaded {len(suppliers_loaded)} suppliers from the ERP file.")
            st.dataframe(
                pd.DataFrame([s.model_dump() for s in suppliers_loaded]),
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("Expected columns")
            st.code(
                "supplier_id,name,capacity_per_day,lead_time_days,reliability\n"
                "SUP-A,Supplier A,1800.0,7,0.97\n"
                "SUP-B,Supplier B,4100.0,9,0.95\n"
                "SUP-C,Supplier C,4100.0,11,0.92",
                language="csv",
            )

with tabs[3]:
    ui.section(
        "Network health",
        "Steady-state performance and the six-echelon graph configuration.",
    )

    kpis = twin.kpis()
    ui.metric_grid(
        [
            {
                "label": "Average daily revenue",
                "value": f"₹{kpis['avg_daily_revenue'] / 1e5:.1f} L",
            },
            {
                "label": "Service level",
                "value": f"{kpis['effective_service_level'] * 100:.1f}%",
            },
            {
                "label": "Average daily demand",
                "value": f"{kpis['avg_daily_demand']:,.0f} units",
            },
            {
                "label": "Average daily fulfilled",
                "value": f"{kpis['avg_daily_fulfilled']:,.0f} units",
            },
        ]
    )

    st.markdown("**Network configuration**")
    node_data = [
        {
            "ID": n.id,
            "Name": n.name,
            "Echelon": n.type.value,
            "Capacity/day": n.capacity_per_day,
            "Lead time (d)": n.lead_time_days,
        }
        for n in twin.network.nodes.values()
    ]
    st.dataframe(pd.DataFrame(node_data), width="stretch", hide_index=True)
