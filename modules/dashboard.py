"""
Module 1: Portfolio Monitoring Dashboard
Fund performance (IRR, TVPI, DPI, RVPI), exposure by strategy/geography/vintage,
cash flow timelines, commitment tracking (funded vs unfunded).
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render(funds: pd.DataFrame, quarterly: pd.DataFrame, cash_flows: pd.DataFrame,
           gps: pd.DataFrame):
    st.header("Portfolio Monitoring Dashboard")

    # Merge fund metadata
    latest_q = (quarterly.sort_values("quarter_end")
                .groupby("fund_id").last().reset_index())
    fund_perf = funds.merge(latest_q, on="fund_id", how="left")
    fund_perf = fund_perf.merge(gps[["gp_id", "gp_name"]], on="gp_id", how="left")

    with st.expander("ℹ️ What are IRR, TVPI, DPI, and RVPI?"):
        st.markdown("""
**IRR** (Internal Rate of Return) is the annualised return accounting for the timing of cash flows.
Computed here using XIRR (Brent's method) from actual dated capital calls and distributions — not approximated.

**TVPI** (Total Value to Paid-In) = (Cumulative Distributions + Current NAV) / Cumulative Paid-In Capital.
Measures total value created per dollar invested. A TVPI of 1.8x means $1.80 of total value for every $1.00 invested.

**DPI** (Distributions to Paid-In) = Cumulative Distributions / Paid-In Capital.
Measures what's actually been returned in cash. DPI of 1.0x means the fund has fully returned investors' capital.

**RVPI** (Residual Value to Paid-In) = Current NAV / Paid-In Capital.
Measures unrealised value still in the fund. TVPI = DPI + RVPI always holds.
""")

    with st.expander("ℹ️ What is the J-curve?"):
        st.markdown("""
PE funds typically show negative or flat returns in years 1-3 because management fees and setup costs
reduce NAV before investments generate returns. Performance inflects positive as portfolio companies
grow and are exited, usually from year 3-4 onward. The pattern of early losses followed by later gains
creates a J-shaped return curve when plotted over time.
""")

    # ── KPI cards ──────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    total_nav = fund_perf["ending_nav_mm"].sum()
    total_committed = fund_perf["total_commitment_mm"].sum()
    total_funded = fund_perf["cumulative_contributions_mm"].sum()
    total_distributed = fund_perf["cumulative_distributions_mm"].sum()
    avg_irr = fund_perf["irr"].mean()

    col1.metric("Total NAV", f"${total_nav:,.0f}M")
    col2.metric("Total Committed", f"${total_committed:,.0f}M")
    col3.metric("Funded", f"${total_funded:,.0f}M")
    col4.metric("Distributed", f"${total_distributed:,.0f}M")
    col5.metric("Avg IRR", f"{avg_irr:.1%}")

    st.divider()

    # ── Fund Performance Table ─────────────────────────────────────
    st.subheader("Fund Performance Summary")

    filters_col1, filters_col2, filters_col3 = st.columns(3)
    with filters_col1:
        strat_filter = st.multiselect("Strategy", fund_perf["strategy"].unique(),
                                       default=fund_perf["strategy"].unique(),
                                       key="dash_strat")
    with filters_col2:
        geo_filter = st.multiselect("Geography", fund_perf["geography"].unique(),
                                     default=fund_perf["geography"].unique(),
                                     key="dash_geo")
    with filters_col3:
        vintage_range = st.slider("Vintage Year",
                                   int(fund_perf["vintage_year"].min()),
                                   int(fund_perf["vintage_year"].max()),
                                   (int(fund_perf["vintage_year"].min()),
                                    int(fund_perf["vintage_year"].max())),
                                   key="dash_vintage")

    mask = ((fund_perf["strategy"].isin(strat_filter)) &
            (fund_perf["geography"].isin(geo_filter)) &
            (fund_perf["vintage_year"].between(*vintage_range)))
    filtered = fund_perf[mask]

    display_cols = ["fund_name", "gp_name", "strategy", "geography", "vintage_year",
                    "fund_size_mm", "total_commitment_mm", "ending_nav_mm", "irr", "tvpi", "dpi", "rvpi"]
    display_df = filtered[display_cols].copy()
    display_df.columns = ["Fund", "GP", "Strategy", "Geography", "Vintage",
                          "Fund Size ($M)", "LP Commitment ($M)", "NAV ($M)", "IRR", "TVPI", "DPI", "RVPI"]

    st.dataframe(
        display_df.style.format({
            "Fund Size ($M)": "{:,.0f}", "LP Commitment ($M)": "{:,.1f}", "NAV ($M)": "{:,.1f}",
            "IRR": "{:.1%}", "TVPI": "{:.2f}x", "DPI": "{:.2f}x", "RVPI": "{:.2f}x",
        }),
        use_container_width=True,
        height=400,
    )

    st.divider()

    # ── Performance Scatter: IRR vs TVPI ───────────────────────────
    left, right = st.columns(2)

    with left:
        st.subheader("IRR vs TVPI by Strategy")
        fig_scatter = px.scatter(
            filtered, x="irr", y="tvpi", color="strategy",
            size="fund_size_mm", hover_name="fund_name",
            labels={"irr": "IRR", "tvpi": "TVPI (x)", "strategy": "Strategy"},
        )
        fig_scatter.update_layout(
            xaxis_tickformat=".0%", height=420,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with right:
        st.subheader("Exposure by Strategy")
        strat_nav = filtered.groupby("strategy")["ending_nav_mm"].sum().reset_index()
        fig_pie = px.pie(strat_nav, values="ending_nav_mm", names="strategy",
                         hole=0.45)
        fig_pie.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Exposure breakdowns ────────────────────────────────────────
    left2, mid2, right2 = st.columns(3)

    with left2:
        st.subheader("By Geography")
        geo_nav = filtered.groupby("geography")["ending_nav_mm"].sum().reset_index()
        fig_geo = px.bar(geo_nav, x="geography", y="ending_nav_mm", color="geography",
                         labels={"ending_nav_mm": "NAV ($M)"})
        fig_geo.update_layout(showlegend=False, height=350,
                              margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_geo, use_container_width=True)

    with mid2:
        st.subheader("By Vintage Year")
        vin_nav = filtered.groupby("vintage_year")["ending_nav_mm"].sum().reset_index()
        fig_vin = px.bar(vin_nav, x="vintage_year", y="ending_nav_mm",
                         labels={"ending_nav_mm": "NAV ($M)", "vintage_year": "Vintage"})
        fig_vin.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_vin, use_container_width=True)

    with right2:
        st.subheader("Funded vs Unfunded")
        funded = filtered["cumulative_contributions_mm"].sum()
        unfunded = filtered["total_commitment_mm"].sum() - funded
        fig_commit = go.Figure(go.Bar(
            x=["Funded", "Unfunded"], y=[funded, max(0, unfunded)],
            marker_color=["#2ecc71", "#e74c3c"],
        ))
        fig_commit.update_layout(height=350, yaxis_title="$M",
                                 margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_commit, use_container_width=True)

    st.divider()

    # ── Cash Flow Timeline ─────────────────────────────────────────
    st.subheader("Aggregate Cash Flow Timeline")
    with st.expander("ℹ️ How to read this chart"):
        st.markdown("""
Red bars (negative) are **capital calls** — money flowing from the LP into the fund.
Green bars (positive) are **distributions** — money flowing back from the fund to the LP.
Net cash flow turns positive when the portfolio starts returning more capital than it's calling.
Early years are dominated by capital calls (investment period); later years by distributions (harvest period).
""")
    cf = cash_flows.copy()
    cf["date"] = pd.to_datetime(cf["date"])
    cf["quarter"] = cf["date"].dt.to_period("Q").astype(str)

    cf_agg = cf.groupby(["quarter", "cf_type"])["amount_mm"].sum().reset_index()
    fig_cf = px.bar(cf_agg, x="quarter", y="amount_mm", color="cf_type",
                    barmode="relative",
                    labels={"amount_mm": "Cash Flow ($M)", "quarter": "Quarter"},
                    color_discrete_map={"Capital Call": "#e74c3c", "Distribution": "#2ecc71"})
    fig_cf.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
    fig_cf.update_xaxes(tickangle=45, dtick=4)
    st.plotly_chart(fig_cf, use_container_width=True)

    # ── Net Cash Flow Cumulative ───────────────────────────────────
    st.subheader("Cumulative Net Cash Flow")
    cf_net = cf.groupby("quarter")["amount_mm"].sum().reset_index()
    cf_net = cf_net.sort_values("quarter")
    cf_net["cumulative"] = cf_net["amount_mm"].cumsum()
    fig_cum = px.area(cf_net, x="quarter", y="cumulative",
                      labels={"cumulative": "Cumulative Net CF ($M)"})
    fig_cum.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
    fig_cum.update_xaxes(tickangle=45, dtick=4)
    st.plotly_chart(fig_cum, use_container_width=True)
