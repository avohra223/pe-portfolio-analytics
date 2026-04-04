"""
Module 3: GP Behavior Prediction Engine
Historical patterns in markups/markdowns, distribution timing, fund life extensions.
Predict NAV trajectory and cash flow timing per fund.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def _compute_gp_metrics(quarterly: pd.DataFrame, funds: pd.DataFrame,
                        gps: pd.DataFrame) -> pd.DataFrame:
    """Compute GP-level behavioral metrics from quarterly data."""
    q = quarterly.merge(funds[["fund_id", "gp_id", "strategy", "vintage_year"]], on="fund_id")
    q = q.merge(gps[["gp_id", "gp_name", "style"]], on="gp_id")
    q["quarter_end"] = pd.to_datetime(q["quarter_end"])
    q = q.sort_values(["fund_id", "quarter_end"])

    # Quarter-over-quarter NAV change rate (markup/markdown)
    q["nav_change_pct"] = q.groupby("fund_id")["ending_nav_mm"].pct_change()

    # Classify as markup or markdown
    q["markup"] = q["nav_change_pct"] > 0.01
    q["markdown"] = q["nav_change_pct"] < -0.01

    # Fund age in quarters
    q["fund_age_q"] = q.groupby("fund_id").cumcount() + 1

    return q


def render(funds: pd.DataFrame, quarterly: pd.DataFrame, gps: pd.DataFrame):
    st.header("GP Behavior Prediction Engine")
    st.caption("Analyze historical GP behavior patterns — markup/markdown tendencies, "
               "distribution timing, and fund life management. Predict future NAV trajectory.")

    q = _compute_gp_metrics(quarterly, funds, gps)

    # ── GP Selector ────────────────────────────────────────────────
    gp_list = gps[["gp_id", "gp_name", "style"]].copy()
    gp_options = dict(zip(gp_list["gp_name"], gp_list["gp_id"]))
    selected_gp_name = st.selectbox("Select GP", list(gp_options.keys()))
    selected_gp = gp_options[selected_gp_name]

    gp_data = q[q["gp_id"] == selected_gp]
    gp_funds = funds[funds["gp_id"] == selected_gp]
    gp_style = gps[gps["gp_id"] == selected_gp]["style"].iloc[0]

    # ── GP Profile ─────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Behavioral Style", gp_style.title())
    k2.metric("Funds Managed", f"{len(gp_funds)}")
    avg_markup_rate = gp_data["markup"].mean()
    k3.metric("Markup Frequency", f"{avg_markup_rate:.1%}")
    avg_markdown_rate = gp_data["markdown"].mean()
    k4.metric("Markdown Frequency", f"{avg_markdown_rate:.1%}")

    st.divider()

    # ── Markup / Markdown Pattern Over Fund Life ───────────────────
    st.subheader("Markup/Markdown Pattern by Fund Age")

    age_pattern = (gp_data.groupby("fund_age_q")
                   .agg(avg_nav_change=("nav_change_pct", "mean"),
                        markup_rate=("markup", "mean"),
                        markdown_rate=("markdown", "mean"))
                   .reset_index())

    fig_pattern = make_subplots(specs=[[{"secondary_y": True}]])
    fig_pattern.add_trace(
        go.Bar(x=age_pattern["fund_age_q"], y=age_pattern["avg_nav_change"],
               name="Avg NAV Change %", marker_color=np.where(
                   age_pattern["avg_nav_change"] >= 0, "#2ecc71", "#e74c3c")),
        secondary_y=False,
    )
    fig_pattern.add_trace(
        go.Scatter(x=age_pattern["fund_age_q"], y=age_pattern["markup_rate"],
                   name="Markup Rate", line=dict(color="#3498db", width=2)),
        secondary_y=True,
    )
    fig_pattern.update_layout(
        height=400, margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Fund Age (Quarters)",
    )
    fig_pattern.update_yaxes(title_text="Avg NAV Change %", tickformat=".1%", secondary_y=False)
    fig_pattern.update_yaxes(title_text="Markup Rate", tickformat=".0%", secondary_y=True)
    st.plotly_chart(fig_pattern, use_container_width=True)

    # ── Distribution Timing Analysis ───────────────────────────────
    st.subheader("Distribution Timing Pattern")

    dist_pattern = (gp_data[gp_data["distributions_mm"] > 0]
                    .groupby("fund_age_q")
                    .agg(avg_dist=("distributions_mm", "mean"),
                         n_events=("distributions_mm", "count"))
                    .reset_index())

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Bar(
        x=dist_pattern["fund_age_q"], y=dist_pattern["avg_dist"],
        name="Avg Distribution ($M)", marker_color="#9b59b6",
    ))
    fig_dist.update_layout(
        height=350, xaxis_title="Fund Age (Quarters)",
        yaxis_title="Avg Distribution ($M)",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    left, right = st.columns(2)

    with left:
        st.subheader("Distribution Start Quarter")
        first_dist = (gp_data[gp_data["distributions_mm"] > 0.1]
                      .groupby("fund_id")["fund_age_q"].min().reset_index())
        first_dist.columns = ["fund_id", "first_dist_q"]
        first_dist = first_dist.merge(gp_funds[["fund_id", "fund_name"]], on="fund_id")
        fig_first = px.bar(first_dist, x="fund_name", y="first_dist_q",
                           labels={"first_dist_q": "Quarter #", "fund_name": ""})
        fig_first.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_first, use_container_width=True)

    with right:
        st.subheader("Cumulative DPI by Fund")
        for _, fund_row in gp_funds.iterrows():
            fdata = gp_data[gp_data["fund_id"] == fund_row["fund_id"]].sort_values("quarter_end")
            if len(fdata) == 0:
                continue

        dpi_data = gp_data[["fund_id", "fund_age_q", "dpi"]].sort_values(["fund_id", "fund_age_q"]).copy()
        dpi_data = dpi_data.merge(gp_funds[["fund_id", "fund_name"]], on="fund_id")
        fig_dpi = px.line(dpi_data, x="fund_age_q", y="dpi", color="fund_name",
                          labels={"dpi": "DPI (x)", "fund_age_q": "Fund Age (Quarters)"})
        fig_dpi.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20),
                              legend=dict(font=dict(size=9)))
        st.plotly_chart(fig_dpi, use_container_width=True)

    st.divider()

    # ── NAV Trajectory Prediction ──────────────────────────────────
    st.subheader("NAV Trajectory Prediction")

    fund_options = dict(zip(gp_funds["fund_name"], gp_funds["fund_id"]))
    if fund_options:
        selected_fund_name = st.selectbox("Select Fund for Prediction",
                                           list(fund_options.keys()))
        selected_fund_id = fund_options[selected_fund_name]
        fund_q = gp_data[gp_data["fund_id"] == selected_fund_id].sort_values("quarter_end")

        if len(fund_q) >= 4:
            X = fund_q["fund_age_q"].values.reshape(-1, 1)
            y = fund_q["ending_nav_mm"].values

            # Polynomial regression for NAV prediction
            poly = PolynomialFeatures(degree=3)
            X_poly = poly.fit_transform(X)
            model = LinearRegression()
            model.fit(X_poly, y)

            # Predict 8 quarters ahead
            max_q = int(X.max())
            future_q = np.arange(1, max_q + 9).reshape(-1, 1)
            future_poly = poly.transform(future_q)
            predicted = model.predict(future_poly)
            predicted = np.maximum(predicted, 0)

            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=fund_q["fund_age_q"], y=fund_q["ending_nav_mm"],
                mode="lines+markers", name="Actual NAV",
                line=dict(color="#2c3e50", width=2),
            ))
            fig_pred.add_trace(go.Scatter(
                x=future_q.flatten(), y=predicted,
                mode="lines", name="Predicted NAV",
                line=dict(color="#e67e22", width=2, dash="dash"),
            ))
            # Confidence band
            residuals = y - model.predict(X_poly)
            std_resid = np.std(residuals)
            fig_pred.add_trace(go.Scatter(
                x=np.concatenate([future_q.flatten(), future_q.flatten()[::-1]]),
                y=np.concatenate([predicted + 1.96 * std_resid,
                                  (predicted - 1.96 * std_resid)[::-1]]),
                fill="toself", fillcolor="rgba(230,126,34,0.15)",
                line=dict(color="rgba(0,0,0,0)"), name="95% CI",
            ))
            fig_pred.add_vline(x=max_q, line_dash="dot", line_color="gray",
                               annotation_text="Forecast Start")
            fig_pred.update_layout(
                height=450, xaxis_title="Fund Age (Quarters)",
                yaxis_title="NAV ($M)",
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_pred, use_container_width=True)

            # Cash flow timing prediction
            st.subheader("Predicted Cash Flow Timing")
            fund_cf = fund_q[["fund_age_q", "contributions_mm", "distributions_mm"]].copy()
            avg_contrib_late = fund_cf[fund_cf["fund_age_q"] > max_q - 4]["contributions_mm"].mean()
            avg_dist_late = fund_cf[fund_cf["fund_age_q"] > max_q - 4]["distributions_mm"].mean()

            future_quarters = list(range(max_q + 1, max_q + 9))
            pred_contribs = [max(0, avg_contrib_late * (0.5 ** (i / 4))) for i in range(8)]
            pred_dists = [avg_dist_late * (1 + 0.1 * i) for i in range(8)]

            pred_cf = pd.DataFrame({
                "Quarter": [f"Q+{i+1}" for i in range(8)],
                "Predicted Contributions ($M)": [round(c, 1) for c in pred_contribs],
                "Predicted Distributions ($M)": [round(d, 1) for d in pred_dists],
            })
            st.dataframe(pred_cf, use_container_width=True, hide_index=True)
        else:
            st.info("Not enough quarterly data for prediction (need 4+ quarters).")
    else:
        st.info("No funds found for this GP.")

    st.divider()

    # ── GP Comparison ──────────────────────────────────────────────
    st.subheader("GP Behavioral Comparison")
    gp_summary = (q.groupby(["gp_id", "gp_name"])
                  .agg(avg_markup=("nav_change_pct", lambda x: (x > 0.01).mean()),
                       avg_markdown=("nav_change_pct", lambda x: (x < -0.01).mean()),
                       avg_nav_change=("nav_change_pct", "mean"),
                       n_quarters=("fund_id", "count"))
                  .reset_index())
    gp_summary = gp_summary.merge(gps[["gp_id", "style"]], on="gp_id")

    fig_comp = px.scatter(
        gp_summary, x="avg_markup", y="avg_markdown", size="n_quarters",
        color="style", hover_name="gp_name",
        labels={"avg_markup": "Markup Frequency", "avg_markdown": "Markdown Frequency"},
    )
    fig_comp.update_layout(
        height=400, xaxis_tickformat=".0%", yaxis_tickformat=".0%",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig_comp, use_container_width=True)
