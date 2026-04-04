"""
Module 6: Macro Stress Test Simulator
Model how rate hikes, recession, sector shocks propagate through portfolio
companies up to fund/portfolio NAV.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Sector sensitivity to macro factors (betas)
SECTOR_SENSITIVITIES = {
    #                        rate_hike  recession  inflation  sector_shock
    "Technology":           [-0.25,     -0.20,     -0.10,     0.00],
    "Healthcare":           [-0.05,     -0.05,      0.05,     0.00],
    "Consumer / Retail":    [-0.15,     -0.30,     -0.20,     0.00],
    "Industrials":          [-0.10,     -0.25,     -0.10,     0.00],
    "Financial Services":   [-0.20,     -0.30,      0.05,     0.00],
    "Energy":               [ 0.05,     -0.15,      0.15,     0.00],
    "Real Estate":          [-0.30,     -0.20,     -0.05,     0.00],
    "Media & Telecom":      [-0.15,     -0.15,     -0.05,     0.00],
}

STRATEGY_SENSITIVITIES = {
    "Buyout":                    {"leverage_factor": 1.5, "recovery_speed": 0.8},
    "Growth Equity":             {"leverage_factor": 1.0, "recovery_speed": 0.7},
    "Venture Capital":           {"leverage_factor": 0.5, "recovery_speed": 0.5},
    "Distressed / Special Sits": {"leverage_factor": 2.0, "recovery_speed": 1.2},
    "Real Estate":               {"leverage_factor": 1.8, "recovery_speed": 0.6},
    "Infrastructure":            {"leverage_factor": 1.3, "recovery_speed": 0.9},
}

SCENARIOS = {
    "Rate Hike (+300bps)": {
        "description": "Central bank raises rates by 300 basis points over 12 months",
        "rate_hike": 0.30, "recession": 0.0, "inflation": 0.10, "sector_shock": 0.0,
    },
    "Mild Recession": {
        "description": "GDP contracts 1-2%, unemployment rises to 6%, credit tightens",
        "rate_hike": 0.0, "recession": 0.25, "inflation": -0.05, "sector_shock": 0.0,
    },
    "Deep Recession": {
        "description": "GDP contracts 4%+, unemployment >8%, severe credit crunch",
        "rate_hike": -0.10, "recession": 0.50, "inflation": -0.10, "sector_shock": 0.0,
    },
    "Stagflation": {
        "description": "Rising rates + recession + persistent inflation",
        "rate_hike": 0.20, "recession": 0.30, "inflation": 0.25, "sector_shock": 0.0,
    },
    "Tech Sector Crash": {
        "description": "Technology sector multiple compression of 40%+",
        "rate_hike": 0.05, "recession": 0.10, "inflation": 0.0, "sector_shock": 0.40,
        "target_sector": "Technology",
    },
    "Energy Crisis": {
        "description": "Energy prices spike 80%+, cascading effects across sectors",
        "rate_hike": 0.10, "recession": 0.15, "inflation": 0.30, "sector_shock": 0.30,
        "target_sector": "Energy",
    },
}


def _apply_stress(holdings: pd.DataFrame, companies: pd.DataFrame,
                  funds: pd.DataFrame, quarterly: pd.DataFrame,
                  scenario: dict) -> pd.DataFrame:
    """Apply a macro stress scenario and return impacted fund NAVs."""

    # Enrich holdings with company and fund data
    h = (holdings
         .merge(companies[["company_id", "sector", "region"]], on="company_id", how="left")
         .merge(funds[["fund_id", "fund_name", "strategy"]], on="fund_id", how="left"))

    # Compute company-level NAV impact
    impacts = []
    for _, row in h.iterrows():
        sector = row["sector"]
        strategy = row["strategy"]
        sensitivities = SECTOR_SENSITIVITIES.get(sector, [0, 0, 0, 0])
        strat_profile = STRATEGY_SENSITIVITIES.get(strategy,
                                                     {"leverage_factor": 1.0, "recovery_speed": 0.8})

        # Base impact from macro factors
        impact = (sensitivities[0] * scenario.get("rate_hike", 0) +
                  sensitivities[1] * scenario.get("recession", 0) +
                  sensitivities[2] * scenario.get("inflation", 0))

        # Sector-specific shock
        target = scenario.get("target_sector")
        if target and sector == target:
            impact -= scenario.get("sector_shock", 0)
        elif target and sector != target:
            # Spillover: ~20% of sector shock affects others
            impact -= scenario.get("sector_shock", 0) * 0.20

        # Amplify by leverage
        impact *= strat_profile["leverage_factor"]

        impacts.append({
            "fund_id": row["fund_id"],
            "fund_name": row["fund_name"],
            "company_id": row["company_id"],
            "sector": sector,
            "strategy": strategy,
            "initial_cost_mm": row["initial_cost_mm"],
            "nav_impact_pct": impact,
            "nav_impact_mm": row["initial_cost_mm"] * impact,
        })

    impact_df = pd.DataFrame(impacts)

    # Aggregate to fund level
    fund_impact = (impact_df.groupby(["fund_id", "fund_name", "strategy"])
                   .agg(total_cost=("initial_cost_mm", "sum"),
                        total_impact_mm=("nav_impact_mm", "sum"),
                        avg_impact_pct=("nav_impact_pct", "mean"))
                   .reset_index())

    # Merge with latest NAV
    latest = (quarterly.sort_values("quarter_end")
              .groupby("fund_id").last()[["ending_nav_mm", "tvpi", "irr"]].reset_index())
    fund_impact = fund_impact.merge(latest, on="fund_id", how="left")
    fund_impact["stressed_nav_mm"] = (fund_impact["ending_nav_mm"] +
                                       fund_impact["total_impact_mm"]).clip(lower=0)
    fund_impact["nav_change_pct"] = (fund_impact["total_impact_mm"] /
                                      fund_impact["ending_nav_mm"].clip(lower=0.01))

    return fund_impact, impact_df


def render(funds: pd.DataFrame, quarterly: pd.DataFrame, companies: pd.DataFrame,
           holdings: pd.DataFrame, gps: pd.DataFrame):
    st.header("Macro Stress Test Simulator")
    st.caption("Model how macro shocks propagate through portfolio companies "
               "to fund and portfolio-level NAV.")

    # ── Scenario Selection ─────────────────────────────────────────
    left_sel, right_sel = st.columns([2, 1])

    with left_sel:
        scenario_name = st.selectbox("Select Macro Scenario", list(SCENARIOS.keys()))
    with right_sel:
        severity_mult = st.slider("Severity Multiplier", 0.5, 2.0, 1.0, 0.1,
                                   help="Scale the scenario intensity")

    scenario = SCENARIOS[scenario_name].copy()
    st.info(f"**Scenario:** {scenario.pop('description')}")

    # Apply severity multiplier
    for key in ["rate_hike", "recession", "inflation", "sector_shock"]:
        if key in scenario:
            scenario[key] *= severity_mult

    # ── Custom Scenario ────────────────────────────────────────────
    with st.expander("Custom Scenario Parameters"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            scenario["rate_hike"] = st.slider("Rate Change", -0.20, 0.50,
                                               float(scenario.get("rate_hike", 0)), 0.05,
                                               key="custom_rate")
        with c2:
            scenario["recession"] = st.slider("Recession Severity", 0.0, 0.80,
                                               float(scenario.get("recession", 0)), 0.05,
                                               key="custom_recession")
        with c3:
            scenario["inflation"] = st.slider("Inflation Shock", -0.20, 0.50,
                                               float(scenario.get("inflation", 0)), 0.05,
                                               key="custom_inflation")
        with c4:
            scenario["sector_shock"] = st.slider("Sector Shock", 0.0, 0.60,
                                                  float(scenario.get("sector_shock", 0)), 0.05,
                                                  key="custom_sector")

    # ── Run Stress Test ────────────────────────────────────────────
    fund_impact, company_impact = _apply_stress(holdings, companies, funds, quarterly, scenario)

    # ── Portfolio KPIs ─────────────────────────────────────────────
    total_base_nav = fund_impact["ending_nav_mm"].sum()
    total_stressed_nav = fund_impact["stressed_nav_mm"].sum()
    total_loss = total_stressed_nav - total_base_nav
    pct_loss = total_loss / total_base_nav if total_base_nav > 0 else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Base Portfolio NAV", f"${total_base_nav:,.0f}M")
    k2.metric("Stressed NAV", f"${total_stressed_nav:,.0f}M")
    k3.metric("NAV Impact", f"${total_loss:+,.0f}M", delta=f"{pct_loss:+.1%}",
              delta_color="inverse")
    worst_fund = fund_impact.loc[fund_impact["nav_change_pct"].idxmin()]
    k4.metric("Worst Hit Fund", f"{worst_fund['fund_name']}",
              delta=f"{worst_fund['nav_change_pct']:+.1%}", delta_color="inverse")

    st.divider()

    # ── Fund-Level Impact ──────────────────────────────────────────
    st.subheader("Fund-Level NAV Impact")

    fund_display = fund_impact.sort_values("nav_change_pct")
    fig_waterfall = go.Figure()
    fig_waterfall.add_trace(go.Bar(
        x=fund_display["fund_name"], y=fund_display["ending_nav_mm"],
        name="Base NAV", marker_color="#3498db", opacity=0.7,
    ))
    fig_waterfall.add_trace(go.Bar(
        x=fund_display["fund_name"], y=fund_display["total_impact_mm"],
        name="Stress Impact", marker_color=np.where(
            fund_display["total_impact_mm"] < 0, "#e74c3c", "#2ecc71"),
    ))
    fig_waterfall.update_layout(
        barmode="relative", height=450,
        yaxis_title="NAV ($M)", xaxis_tickangle=45,
        margin=dict(l=20, r=20, t=30, b=100),
    )
    st.plotly_chart(fig_waterfall, use_container_width=True)

    # ── Impact by Strategy ─────────────────────────────────────────
    left, right = st.columns(2)

    with left:
        st.subheader("Impact by Strategy")
        strat_impact = (fund_impact.groupby("strategy")
                        .agg(base_nav=("ending_nav_mm", "sum"),
                             stressed_nav=("stressed_nav_mm", "sum"),
                             avg_impact=("nav_change_pct", "mean"))
                        .reset_index()
                        .sort_values("avg_impact"))
        fig_strat = px.bar(strat_impact, x="strategy", y="avg_impact",
                           color="avg_impact", color_continuous_scale="RdYlGn",
                           labels={"avg_impact": "Avg NAV Impact %", "strategy": ""})
        fig_strat.update_layout(
            height=400, yaxis_tickformat=".1%",
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_strat, use_container_width=True)

    with right:
        st.subheader("Impact by Sector")
        sector_impact = (company_impact.groupby("sector")
                         .agg(total_impact=("nav_impact_mm", "sum"),
                              avg_impact=("nav_impact_pct", "mean"))
                         .reset_index()
                         .sort_values("avg_impact"))
        fig_sec = px.bar(sector_impact, y="sector", x="avg_impact", orientation="h",
                         color="avg_impact", color_continuous_scale="RdYlGn",
                         labels={"avg_impact": "Avg Impact %", "sector": ""})
        fig_sec.update_layout(
            height=400, xaxis_tickformat=".1%",
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_sec, use_container_width=True)

    st.divider()

    # ── Scenario Comparison ────────────────────────────────────────
    st.subheader("Scenario Comparison Matrix")
    comparison = []
    for sname, sconfig in SCENARIOS.items():
        sc = {k: v for k, v in sconfig.items() if k not in ["description", "target_sector"]}
        if "target_sector" in sconfig:
            sc["target_sector"] = sconfig["target_sector"]
        fi, _ = _apply_stress(holdings, companies, funds, quarterly, sc)
        base = fi["ending_nav_mm"].sum()
        stressed = fi["stressed_nav_mm"].sum()
        comparison.append({
            "Scenario": sname,
            "Base NAV ($M)": base,
            "Stressed NAV ($M)": stressed,
            "Impact ($M)": stressed - base,
            "Impact (%)": (stressed - base) / base if base > 0 else 0,
        })
    comp_df = pd.DataFrame(comparison)

    fig_comp = px.bar(comp_df.sort_values("Impact (%)"), x="Impact (%)", y="Scenario",
                      orientation="h", color="Impact (%)",
                      color_continuous_scale="RdYlGn")
    fig_comp.update_layout(
        height=400, xaxis_tickformat=".1%",
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    st.dataframe(
        comp_df.style.format({
            "Base NAV ($M)": "{:,.0f}",
            "Stressed NAV ($M)": "{:,.0f}",
            "Impact ($M)": "{:+,.0f}",
            "Impact (%)": "{:+.2%}",
        }),
        use_container_width=True,
    )

    st.divider()

    # ── Detailed Fund Impact Table ─────────────────────────────────
    st.subheader("Detailed Fund Impact")
    detail_df = fund_impact[["fund_name", "strategy", "ending_nav_mm",
                              "stressed_nav_mm", "total_impact_mm", "nav_change_pct"]].copy()
    detail_df.columns = ["Fund", "Strategy", "Base NAV ($M)", "Stressed NAV ($M)",
                         "Impact ($M)", "Impact (%)"]
    detail_df = detail_df.sort_values("Impact (%)")

    st.dataframe(
        detail_df.style.format({
            "Base NAV ($M)": "{:,.1f}",
            "Stressed NAV ($M)": "{:,.1f}",
            "Impact ($M)": "{:+,.1f}",
            "Impact (%)": "{:+.2%}",
        }),
        use_container_width=True, height=500,
    )
