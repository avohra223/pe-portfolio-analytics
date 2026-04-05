"""
Module 4: Secondary Pricing Model
Estimate bid price as % of NAV based on fund age, strategy, cash flow patterns,
GP track record, unfunded commitments.

Uses Ridge regression (better than gradient boosting for N=30) with
Leave-One-Out cross-validation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler


def _build_pricing_features(funds: pd.DataFrame, quarterly: pd.DataFrame,
                            gps: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix for secondary pricing model. All 30 funds get a price."""
    latest = (quarterly.sort_values("quarter_end")
              .groupby("fund_id").last().reset_index())
    df = funds.merge(latest, on="fund_id", how="left")
    df = df.merge(gps[["gp_id", "gp_name", "track_record_score", "style"]], on="gp_id", how="left")

    # Fill any NaN values from the merge (shouldn't happen, but defensive)
    df["ending_nav_mm"] = df["ending_nav_mm"].fillna(0)
    df["tvpi"] = df["tvpi"].fillna(1.0)
    df["dpi"] = df["dpi"].fillna(0)
    df["rvpi"] = df["rvpi"].fillna(1.0)
    df["irr"] = df["irr"].fillna(0)
    df["cumulative_contributions_mm"] = df["cumulative_contributions_mm"].fillna(
        df["total_commitment_mm"] * 0.5)
    df["track_record_score"] = df["track_record_score"].fillna(0.7)

    # Derived features
    df["fund_age_years"] = (2025 - df["vintage_year"]).clip(lower=1)
    df["remaining_life_years"] = (df["fund_term_years"] - df["fund_age_years"]).clip(lower=0)
    df["pct_funded"] = (df["cumulative_contributions_mm"] /
                        df["total_commitment_mm"].clip(lower=0.01)).clip(upper=1.0)
    df["unfunded_ratio"] = 1 - df["pct_funded"]

    # Strategy encoding — ordinal score based on secondary market liquidity
    strat_map = {
        "Buyout": 1.0, "Growth Equity": 0.95, "Venture Capital": 0.85,
        "Distressed / Special Sits": 0.88, "Real Estate": 0.90, "Infrastructure": 0.93,
    }
    df["strategy_score"] = df["strategy"].map(strat_map).fillna(0.9)

    # ── Generate synthetic secondary price (the "market price" we're modeling) ──
    # Formula with wider coefficient spread so the model has a real signal to learn
    rng = np.random.default_rng(99)

    df["secondary_price_pct"] = (
        0.75                                                # base discount
        + 0.08 * df["dpi"].clip(upper=2.0)                  # high DPI → closer to NAV
        - 0.08 * df["unfunded_ratio"]                        # unfunded liability discount
        + 0.12 * df["track_record_score"]                    # GP quality premium
        + 0.04 * df["tvpi"].clip(upper=3.0)                  # performance premium
        + 0.06 * df["strategy_score"]                        # strategy liquidity
        - 0.005 * df["remaining_life_years"].clip(upper=8)   # illiquidity discount
        + 0.015 * df["fund_age_years"].clip(upper=10)        # mature funds priced tighter
        - 0.10 * (df["irr"] < 0).astype(float)              # negative IRR penalty
        + rng.normal(0, 0.025, len(df))                      # market noise
    ).clip(0.50, 1.20).round(4)

    return df


def render(funds: pd.DataFrame, quarterly: pd.DataFrame, gps: pd.DataFrame):
    st.header("Secondary Pricing Model")
    st.caption("Estimate secondary market bid price as % of NAV using fund characteristics, "
               "GP track record, and market dynamics.")

    df = _build_pricing_features(funds, quarterly, gps)
    n_funds = len(df)

    # ── Feature matrix ─────────────────────────────────────────────
    feature_cols = ["fund_age_years", "remaining_life_years", "tvpi", "dpi",
                    "unfunded_ratio", "track_record_score", "strategy_score", "irr"]
    X = df[feature_cols].fillna(0).values
    y = df["secondary_price_pct"].values

    # Scale features for Ridge
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Ridge regression — much better than gradient boosting for N=30
    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y)

    # Leave-One-Out CV — compute R² from LOO predictions vs actuals
    loo = LeaveOneOut()
    loo_predictions = cross_val_predict(model, X_scaled, y, cv=loo).clip(0.50, 1.20)
    df["predicted_price"] = loo_predictions

    # R² from LOO predictions: 1 - SS_res/SS_tot
    ss_res = np.sum((y - loo_predictions) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    cv_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    df["pricing_gap"] = df["predicted_price"] - df["secondary_price_pct"]

    # ── KPIs ───────────────────────────────────────────────────────
    n_premium = (df["secondary_price_pct"] > 1.01).sum()
    n_discount = (df["secondary_price_pct"] < 0.99).sum()
    n_par = n_funds - n_premium - n_discount

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Avg Secondary Price", f"{df['secondary_price_pct'].mean():.1%} of NAV")
    k2.metric("Model R² (LOO-CV)", f"{cv_r2:.3f}")
    k3.metric("Funds at Premium", f"{n_premium}")
    k4.metric("Funds at Par", f"{n_par}")
    k5.metric("Funds at Discount", f"{n_discount}")

    st.divider()

    # ── Pricing Distribution ───────────────────────────────────────
    st.subheader("Secondary Price Distribution (% of NAV)")
    fig_hist = px.histogram(df, x="secondary_price_pct", nbins=15, color="strategy",
                            labels={"secondary_price_pct": "Price (% of NAV)"},
                            barmode="overlay", opacity=0.7)
    fig_hist.add_vline(x=1.0, line_dash="dot", line_color="red",
                       annotation_text="NAV Par")
    fig_hist.update_layout(height=400, xaxis_tickformat=".0%",
                           margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── Actual vs Predicted ────────────────────────────────────────
    st.subheader("Model Fit: Actual vs Predicted Price")
    fig_fit = px.scatter(df, x="secondary_price_pct", y="predicted_price",
                         color="strategy", hover_name="fund_name",
                         labels={"secondary_price_pct": "Market Price (% NAV)",
                                 "predicted_price": "Model Price (% NAV)"})
    # Perfect prediction line
    price_range = [df["secondary_price_pct"].min() - 0.02, df["secondary_price_pct"].max() + 0.02]
    fig_fit.add_trace(go.Scatter(x=price_range, y=price_range, mode="lines",
                                  line=dict(color="gray", dash="dash"), name="Perfect Fit"))
    fig_fit.update_layout(height=400, xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                          margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_fit, use_container_width=True)

    # ── Pricing by Strategy & Vintage ──────────────────────────────
    left, right = st.columns(2)

    with left:
        st.subheader("Price by Strategy")
        strat_price = (df.groupby("strategy")["secondary_price_pct"]
                       .agg(["mean", "std"]).reset_index())
        strat_price.columns = ["Strategy", "Avg Price", "Std Dev"]
        strat_price["Std Dev"] = strat_price["Std Dev"].fillna(0)
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

    # ── Feature Importance (Ridge coefficients) ────────────────────
    st.subheader("Pricing Model — Feature Coefficients")
    st.caption("Ridge regression coefficients (standardised). Positive = increases price, "
               "negative = decreases price.")
    coef_df = pd.DataFrame({
        "Feature": feature_cols,
        "Coefficient": model.coef_,
    }).sort_values("Coefficient", ascending=True)

    fig_coef = px.bar(coef_df, y="Feature", x="Coefficient", orientation="h",
                      color="Coefficient", color_continuous_scale="RdYlGn",
                      color_continuous_midpoint=0)
    fig_coef.update_layout(height=400, coloraxis_showscale=False,
                           margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_coef, use_container_width=True)

    st.divider()

    # ── Fund-Level Pricing Table ───────────────────────────────────
    st.subheader("Fund-Level Secondary Pricing")
    pricing_table = df[["fund_name", "gp_name", "strategy", "vintage_year",
                        "ending_nav_mm", "tvpi", "dpi", "unfunded_ratio",
                        "track_record_score", "secondary_price_pct", "predicted_price"]].copy()
    pricing_table["implied_value_mm"] = (pricing_table["ending_nav_mm"] *
                                          pricing_table["secondary_price_pct"])

    # Confidence flag for young funds
    pricing_table["confidence"] = np.where(
        pricing_table["vintage_year"] >= 2023, "Low",
        np.where(pricing_table["vintage_year"] >= 2021, "Medium", "High"))

    pricing_table.columns = ["Fund", "GP", "Strategy", "Vintage", "NAV ($M)", "TVPI",
                             "DPI", "Unfunded %", "GP Score", "Market Price", "Model Price",
                             "Implied Value ($M)", "Confidence"]
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
             "Distressed / Special Sits": 0.88, "Real Estate": 0.90,
             "Infrastructure": 0.93}.keys()), key="sim_strat_sel")
        strat_scores = {"Buyout": 1.0, "Growth Equity": 0.95, "Venture Capital": 0.85,
                        "Distressed / Special Sits": 0.88, "Real Estate": 0.90,
                        "Infrastructure": 0.93}

    sim_raw = np.array([[sim_age, sim_remaining, sim_tvpi, sim_dpi,
                          sim_unfunded, sim_gp_score, strat_scores[sim_strat], sim_irr]])
    sim_scaled = scaler.transform(sim_raw)
    sim_price = float(model.predict(sim_scaled)[0])
    sim_price = np.clip(sim_price, 0.50, 1.20)

    st.metric("Estimated Secondary Price", f"{sim_price:.1%} of NAV",
              delta=f"{sim_price - 1.0:+.1%} vs par")
