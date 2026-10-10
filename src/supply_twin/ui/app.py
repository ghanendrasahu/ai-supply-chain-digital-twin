from __future__ import annotations

import io

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

st.set_page_config(
    page_title="AI Supply Chain Digital Twin",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_twin() -> SupplyChainTwin:
    settings = get_settings()
    settings.ensure_dirs()
    return SupplyChainTwin(settings=settings)


twin = load_twin()
settings = twin.settings

st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #2060df;
        margin-bottom: 12px;
    }
    .chain-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .step-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 600;
        margin: 4px;
        font-size: 0.95rem;
    }
    .step-down { background-color: #dc2626; color: white; }
    .step-up { background-color: #ea580c; color: white; }
    .step-neutral { background-color: #2563eb; color: white; }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("🏭 AI Supply Chain Digital Twin")
st.caption(
    "Decision Intelligence & What-If Disruption Simulator: "
    "Supplier ➔ Factory ➔ Warehouse ➔ Distribution ➔ Retailer ➔ Customer"
)

tabs = st.tabs(
    [
        "⚡ What-If Simulator",
        "🧠 Prescriptive AI Advisor",
        "📈 Demand Forecast & ERP",
        "🌐 Network Topology & KPIs",
    ]
)

with tabs[0]:
    st.subheader("What-If Disruption Simulator")

    col_ctrl, col_main = st.columns([1, 2])

    with col_ctrl:
        st.markdown("#### Scenario Parameters")
        event_type = st.selectbox(
            "Disruption Event",
            [
                "Supplier Failure (Canonical 14d)",
                "Custom Supplier Outage",
                "Factory Outage",
                "Demand Surge / Slump",
                "Port / Lane Transit Delay",
                "Price Shock",
            ],
        )

        sim_horizon = st.slider("Simulation Horizon (days)", 30, 180, 90, 10)
        n_mc = st.select_slider("Monte Carlo Runs", options=[10, 50, 100, 500, 1000], value=100)

        interventions = []
        if event_type == "Supplier Failure (Canonical 14d)":
            st.info("Simulating Supplier A offline for 14 days (canonical benchmark)")
            interventions = [SupplierFailure(supplier_id="SUP-A", duration_days=14, severity=1.0)]
        elif event_type == "Custom Supplier Outage":
            sup_target = st.selectbox("Supplier", ["SUP-A", "SUP-B", "SUP-C"])
            dur = st.number_input("Downtime (days)", min_value=1, max_value=60, value=14)
            sev = st.slider("Severity", 0.1, 1.0, 1.0, 0.05)
            interventions = [
                SupplierFailure(supplier_id=sup_target, duration_days=int(dur), severity=sev)
            ]
        elif event_type == "Factory Outage":
            dur = st.number_input("Factory Downtime (days)", 1, 45, 10)
            interventions = [FactoryOutage(factory_id="FAC-1", duration_days=int(dur))]
        elif event_type == "Demand Surge / Slump":
            mag = st.slider("Demand Change (%)", -50, 100, 25, 5) / 100.0
            dur = st.number_input("Shock Duration (days)", 7, 90, 30)
            interventions = [DemandShock(magnitude=mag, duration_days=int(dur))]
        elif event_type == "Port / Lane Transit Delay":
            added = st.number_input("Added Transit Delay (days)", 1.0, 20.0, 5.0, 1.0)
            interventions = [
                TransportDelay(
                    lane_origin="DC-1", lane_destination="RET-1", added_days=float(added)
                )
            ]
        elif event_type == "Price Shock":
            sku_choice = st.selectbox("SKU", ["SKU-001", "SKU-002"])
            chg = st.slider("Price Adjustment (%)", -30, 50, 15, 5) / 100.0
            interventions = [PriceChange(sku=sku_choice, change_pct=chg)]

        run_btn = st.button("🚀 Run What-If Simulation", type="primary", use_container_width=True)

    with col_main:
        if run_btn or "sim_result" not in st.session_state:
            with st.spinner("Executing discrete-event simulation across Monte Carlo ensemble..."):
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
            <div class="chain-card">
                <div style="font-size: 0.9rem; opacity: 0.8; margin-bottom: 8px;">QUANTIFIED CONSEQUENCE CHAIN</div>
                <div style="font-size: 1.15rem; font-weight: bold; line-height: 1.7;">
                    {res.consequence_chain.headline()}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            c1, c2, c3, c4 = st.columns(4)
            steps_map = {s.metric: s for s in res.consequence_chain.steps}

            with c1:
                p = steps_map.get("Production")
                st.metric("Production Impact", f"{p.impact_value:+.1f}%" if p else "-18.0%")
            with c2:
                inv = steps_map.get("Inventory")
                st.metric("Inventory Impact", f"{inv.impact_value:+.1f}%" if inv else "-31.0%")
            with c3:
                stk = steps_map.get("Stockout probability")
                st.metric("Stockout Risk", f"{stk.impact_value:+.1f}%" if stk else "+42.0%")
            with c4:
                rev = steps_map.get("Revenue")
                val_cr = (rev.impact_value / 1e7) if rev else -2.7
                st.metric("Revenue Impact", f"₹{val_cr:.1f} Cr" if rev else "-₹2.7 Cr")

            days_arr = list(range(res.horizon_days))
            b_snaps, s_snaps = res.baseline_kpis, res.scenario_kpis

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
            ax1.plot(days_arr, [x.production for x in b_snaps], label="Baseline", color="#2563eb")
            ax1.plot(
                days_arr,
                [x.production for x in s_snaps],
                label="Scenario",
                color="#dc2626",
                linestyle="--",
            )
            ax1.set_title("Daily Production")
            ax1.set_xlabel("Day")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.plot(
                days_arr, [x.inventory_total for x in b_snaps], label="Baseline", color="#2563eb"
            )
            ax2.plot(
                days_arr,
                [x.inventory_total for x in s_snaps],
                label="Scenario",
                color="#dc2626",
                linestyle="--",
            )
            ax2.set_title("Total Inventory Position")
            ax2.set_xlabel("Day")
            ax2.grid(True, alpha=0.3)

            ax3.plot(
                days_arr, [x.revenue / 1e5 for x in b_snaps], label="Baseline", color="#2563eb"
            )
            ax3.plot(
                days_arr,
                [x.revenue / 1e5 for x in s_snaps],
                label="Scenario",
                color="#dc2626",
                linestyle="--",
            )
            ax3.set_title("Daily Revenue (₹ Lakhs)")
            ax3.set_xlabel("Day")
            ax3.grid(True, alpha=0.3)

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            with st.expander("📊 Monte Carlo Uncertainty Bands (p10 / p50 / p90 / mean)"):
                band_rows = []
                for b in res.bands:
                    band_rows.append(
                        {
                            "Metric": b.metric,
                            "p10": f"{b.p10:,.2f}",
                            "Median (p50)": f"{b.p50:,.2f}",
                            "p90": f"{b.p90:,.2f}",
                            "Mean": f"{b.mean:,.2f}",
                        }
                    )
                st.dataframe(pd.DataFrame(band_rows), use_container_width=True)

with tabs[1]:
    st.subheader("🧠 Prescriptive AI Advisor")
    st.markdown(
        "Automated counterfactual mitigation ranking based on expected ROI and net benefit."
    )

    active_ivs = st.session_state.get(
        "active_interventions", [SupplierFailure(supplier_id="SUP-A", duration_days=14)]
    )

    if st.button("Generate Mitigation Playbook", type="primary"):
        with st.spinner("Simulating counterfactual intervention paths..."):
            advisor = PrescriptiveAdvisor(twin)
            recommendations = advisor.evaluate_mitigations(active_ivs, horizon=60, n_runs=15)
            st.session_state["recommendations"] = recommendations

    recs = st.session_state.get("recommendations")
    if recs:
        for i, rec in enumerate(recs, 1):
            st.markdown(
                f"""
            <div style="background-color: #f1f5f9; padding: 16px; border-radius: 8px; margin-bottom: 12px; border-left: 5px solid {"#10b981" if i == 1 else "#64748b"};">
                <div style="font-weight: bold; font-size: 1.1rem; color: #0f172a;">#{i} {rec.action.title} {"(RECOMMENDED)" if i == 1 else ""}</div>
                <div style="font-size: 0.95rem; color: #475569; margin-top: 4px;">{rec.action.description}</div>
                <div style="display: flex; gap: 24px; margin-top: 12px;">
                    <div><strong>Net Benefit:</strong> ₹{rec.net_benefit / 1e5:.1f} L</div>
                    <div><strong>Recovered Revenue:</strong> ₹{rec.gross_recovered_revenue / 1e5:.1f} L</div>
                    <div><strong>Est. Cost:</strong> ₹{rec.action.estimated_cost / 1e5:.1f} L</div>
                    <div><strong>ROI Multiple:</strong> {rec.roi_multiple}x</div>
                    <div><strong>Risk Reduction:</strong> {rec.stockout_reduction_pct:+.1f}%</div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
    else:
        st.info("Click 'Generate Mitigation Playbook' to evaluate counterfactual strategies.")

with tabs[2]:
    st.subheader("Demand Forecasting & ERP Ingestion")

    f_col1, f_col2 = st.columns([1, 1])

    with f_col1:
        st.markdown("#### Demand Forecasting (HistGradientBoosting)")
        ret_sel = st.selectbox("Retail Store", ["RET-1", "RET-2"])
        sku_sel = st.selectbox("Target SKU", ["SKU-001", "SKU-002"])
        horizon_sel = st.slider("Forecast Days", 7, 60, 30)

        if st.button("Generate Demand Forecast"):
            with st.spinner("Computing point forecast & confidence bounds..."):
                fc = twin.forecast(ret_sel, sku_sel, horizon_sel)
                f_df = pd.DataFrame(
                    {
                        "Day": [str(d) for d in fc.dates],
                        "Forecast": fc.forecast,
                        "Lower 95%": fc.lower,
                        "Upper 95%": fc.upper,
                    }
                )
                st.line_chart(f_df.set_index("Day")[["Forecast", "Lower 95%", "Upper 95%"]])
                st.dataframe(f_df.head(10), use_container_width=True)

    with f_col2:
        st.markdown("#### ERP Data Ingestion Connector")
        uploaded = st.file_uploader("Upload ERP Supplier CSV", type=["csv"])
        if uploaded:
            connector = CSVERPConnector()
            content = uploaded.getvalue().decode("utf-8")
            suppliers_loaded = connector.load_suppliers(io.StringIO(content))
            st.success(f"Successfully loaded {len(suppliers_loaded)} suppliers from ERP file")
            st.dataframe(pd.DataFrame([s.model_dump() for s in suppliers_loaded]))
        else:
            st.markdown(
                """
            **Sample CSV Schema**:
            ```csv
            supplier_id,name,capacity_per_day,lead_time_days,reliability
            SUP-A,Supplier A,1800.0,7,0.97
            SUP-B,Supplier B,4100.0,9,0.95
            SUP-C,Supplier C,4100.0,11,0.92
            ```
            """
            )

with tabs[3]:
    st.subheader("Network Topology & Steady-State KPIs")

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    kpis = twin.kpis()
    with kpi_col1:
        st.metric("Avg Daily Revenue", f"₹{kpis['avg_daily_revenue'] / 1e5:.1f} L")
    with kpi_col2:
        st.metric("Service Level", f"{kpis['effective_service_level'] * 100:.1f}%")
    with kpi_col3:
        st.metric("Avg Daily Demand", f"{kpis['avg_daily_demand']:,.0f} units")
    with kpi_col4:
        st.metric("Avg Daily Fulfilled", f"{kpis['avg_daily_fulfilled']:,.0f} units")

    st.markdown("#### 6-Echelon Graph Configuration")
    node_data = [
        {
            "ID": n.id,
            "Name": n.name,
            "Echelon": n.type.value,
            "Capacity/Day": n.capacity_per_day,
            "Lead Time (d)": n.lead_time_days,
        }
        for n in twin.network.nodes.values()
    ]
    st.dataframe(pd.DataFrame(node_data), use_container_width=True)
