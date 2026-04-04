"""
Module 4: Secondary Pricing Model
Estimate bid price as % of NAV based on fund age, strategy, cash flow patterns,
GP track record, unfunded commitments.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score


def _build_pricing_features(funds: pd.DataFrame, quarterly: pd.DataFrame,
                            gps: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix for secondary pricing model."""
    latest = (quarterly.sort_values("quarter_end")
              .groupby("fund_id").last().reset_index())
    df = funds.merge(latest, on="fund_id", how="left")
    df = df.merge(gps[["gp_id", "gp_name", "track_record_score", "style"]], on="gp_id", how="left")

    # Features
    df["fund_age_years"] = (2025 - df["vintage_year"]).clip(lower=1)
    df["remaining_life_years"] = (df["fund_term_years"] - df["fund_age_years"]).clip(lower=0)
    df["pct_funded"] = df["cumulative_contributions_mm"] / df["total_commitment_mm"].clip(lower=0.01)
    df["unfunded_ratio"] = 1 - df["pct_funded"].clip(upper=1)
    df["nav_to_commitment"] = df["ending_nav_mm"] / df["total_commitment_mm"].clip(lower=0.01)

    # Strategy encoding
    strat_map = {
        "Buyout": 1.0, "Growth Equity": 0.95, "Venture Capital": 0.85,
        "Distressed / Special Sits": 0.90, "Real Estate": 0.92, "Infrastructure": 0.95,
    }
    df["strategy_score"] = df["strategy"].map(strat_map).fillna(0.9)

    # Synthetic "fair" secondary price as % of NAV
    # Based on PE secondaries market dynamics
    rng = np.random.default_rng(99)
    base_price = 0.90  # baseline ~90% of NAV

    adjustments = (
        + 0.05 * df["dpi"].clip(upper=2)              # Higher DPI → closer to NAV
        - 0.03 * df["unfunded_ratio"]                   # Unfunded liability discount
        + 0.08 * df["track_record_score"]               # GP quality premium
        + 0.02 * df["tvpi"].clip(upper=3)               # Performance premium
        - 0.01 * df["fund_age_years"].clip(upper=12)    # Older funds slight discount
        + 0.05 * df["strategy_score"]                   # Strategy premium
        - 0.02 * df["remaining_life_years"].clip(upper=5)  # Longer remaining = illiquidity discount
    )
    df["secondary_price_pct"] = (base_price + adjustments
                                  + rng.normal(0, 0.02, len(df))).clip(0.50, 1.20)
    df["secondary_price_pct"] = df["secondary_price_pct"].round(4)

    return df


def render(funds: pd.DataFrame, quarterly: pd.DataFrame, gps: pd.DataFrame):
    st.header("Secondary Pricing Model")
    st.caption("Estimate secondary market bid price as % of NAV using fund characteristics, "
               "GP track record, and market dynamics.")

    df = _build_pricing_features(funds, quarterly, gps)

    # ── Train pricing model ────────────────────────────────────────
    feature_cols = ["fund_age_years", "remaining_life_years", "tvpi", "dpi", "rvpi",
                    "unfunded_ratio", "track_record_score", "strategy_score",
                    "nav_to_commitment", "irr"]
    X = df[feature_cols].fillna(0)
    y = df["secondary_price_pct"]

    model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X, y)

    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    df["predicted_price"] = model.predict(X).clip(0.50, 1.20)
    df["pricing_gap"] = df["predicted_price"] - df["secondary_price_pct"]

    # ── KPIs ───────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Avg Secondary Price", f"{df['secondary_price_pct'].mean():.1%} of NAV")
    k2.metric("Model R-squared", f"{cv_scores.mean():.3f}")
    k3.metric("Funds at Premium", f"{(df['secondary_price_pct'] > 1.0).sum()}")
    k4.metric("Funds at Discount", f"{(df['secondary_price_pct'] < 0.9).sum()}")

    st.divider()

    # ── Pricing Distribution ───────────────────────────────────────
    st.subheader("Secondary Price Distribution (% of NAV)")
    fig_hist = px.histogram(df, x="secondary_price_pct", nbins=20, color="strategy",
                            labels={"secondary_price_pct": "Price (% of NAV)"},
                            barmode="overlay", opacity=0.7)
    fig_hist.add_vline(x=1.0, line_dash="dot", line_color="red",
                       annotation_text="NAV Par")
    fig_hist.update_layout(height=400, xaxis_tickformat=".0%",
                           margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── Pricing by Strategy & Vintage ──────────────────────────────
    left, right = st.columns(2)

    with left:
        st.subheader("Price by Strategy")
        strat_price = (df.groupby("strategy")["secondary_price_pct"]
                       .agg(["mean", "std"]).reset_index())
        strat_price.columns = ["Strategy", "Avg Price", "Std Dev"]
        fig_strat = go.Figure()
        fig_strat.add_trace(go.Bar(
            x=strat_price["Strategy"], y=strat_price["Avg Price"],
            error_y=dict(type="data", array=strat_price["Std Dev"]),
            marker_color="#3498db",
        ))
        fig_strat.add_hline(y=1.0, line_dash="dot", line_color="red")
        fig_strat.update_layout(
            height=400, yaxis_title="Avg Price (% of NAV)",
            yaxis_tickformat=".0%",
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_strat, use_container_width=True)

    with right:
        st.subheader("Price by Vintage Year")
        vin_price = df.groupby("vintage_year")["secondary_price_pct"].mean().reset_index()
        fig_vin = px.line(vin_price, x="vintage_year", y="secondary_price_pct",
                          markers=True,
                          labels={"secondary_price_pct": "Avg Price (% of NAV)",
                                  "vintage_year": "Vintage"})
        fig_vin.add_hline(y=1.0, line_dash="dot", line_color="red")
        fig_vin.update_layout(height=400, yaxis_tickformat=".0%",
                              margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_vin, use_container_width=True)

    st.divider()

    # ── Feature Importance ─────────────────────────────────────────
    st.subheader("Pricing Model — Feature Importance")
    importance = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=True)

    fig_imp = px.bar(importance, y="Feature", x="Importance", orientation="h",
                     color="Importance", color_continuous_scale="Viridis")
    fig_imp.update_layout(height=400, coloraxis_showscale=False,
                          margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_imp, use_container_width=True)

    st.divider()

    # ── Fund-Level Pricing Table ───────────────────────────────────
    st.subheader("Fund-Level Secondary Pricing")
    pricing_table = df[["fund_name", "gp_name", "strategy", "vintage_year",
                        "ending_nav_mm", "tvpi", "dpi", "unfunded_ratio",
                        "track_record_score", "secondary_price_pct", "predicted_price"]].copy()
    pricing_table["implied_value_mm"] = pricing_table["ending_nav_mm"] * pricing_table["secondary_price_pct"]
    pricing_table.columns = ["Fund", "GP", "Strategy", "Vintage", "NAV ($M)", "TVPI",
                             "DPI", "Unfunded %", "GP Score", "Market Price", "Model Price",
                             "Implied Value ($M)"]
    pricing_table = pricing_table.sort_values("Market Price", ascending=False)

    st.dataframe(
        pricing_table.style.format({
            "NAV ($M)": "{:,.1f}", "TVPI": "{:.2f}x", "DPI": "{:.2f}x",
            "Unfunded %": "{:.1%}", "GP Score": "{:.2f}",
            "Market Price": "{:.1%}", "Model Price": "{:.1%}",
            "Implied Value ($M)": "{:,.1f}",
        }),
        use_container_width=True, height=500,
    )

    st.divider()

    # ── Interactive Pricing Simulator ──────────────────────────────
    st.subheader("Interactive Pricing Simulator")
    st.caption("Adjust fund parameters to estimate secondary market price.")

    sim_col1, sim_col2, sim_col3 = st.columns(3)
    with sim_col1:
        sim_tvpi = st.slider("TVPI", 0.5, 3.0, 1.5, 0.1, key="sim_tvpi")
        sim_dpi = st.slider("DPI", 0.0, 2.0, 0.5, 0.1, key="sim_dpi")
        sim_irr = st.slider("IRR", -0.20, 0.40, 0.12, 0.01, key="sim_irr")
    with sim_col2:
        sim_age = st.slider("Fund Age (Years)", 1, 15, 6, key="sim_age")
        sim_remaining = st.slider("Remaining Life (Years)", 0, 8, 4, key="sim_remaining")
        sim_unfunded = st.slider("Unfunded Ratio", 0.0, 1.0, 0.3, 0.05, key="sim_unfunded")
    with sim_col3:
        sim_gp_score = st.slider("GP Track Record Score", 0.5, 1.0, 0.75, 0.05, key="sim_gp")
        sim_strat = st.selectbox("Strategy", list(
            {"Buyout": 1.0, "Growth Equity": 0.95, "Venture Capital": 0.85,
             "Distressed / Special Sits": 0.90, "Real Estate": 0.92,
             "Infrastructure": 0.95}.keys()), key="sim_strat_sel")
        strat_scores = {"Buyout": 1.0, "Growth Equity": 0.95, "Venture Capital": 0.85,
                        "Distressed / Special Sits": 0.90, "Real Estate": 0.92,
                        "Infrastructure": 0.95}

    sim_features = np.array([[sim_age, sim_remaining, sim_tvpi, sim_dpi,
                               sim_tvpi - sim_dpi, sim_unfunded, sim_gp_score,
                               strat_scores[sim_strat],
                               sim_tvpi * 0.5, sim_irr]])
    sim_price = model.predict(sim_features)[0]
    sim_price = np.clip(sim_price, 0.50, 1.20)

    st.metric("Estimated Secondary Price", f"{sim_price:.1%} of NAV",
              delta=f"{sim_price - 1.0:+.1%} vs par")
