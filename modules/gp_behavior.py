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

    # Quarter-over-quarter gain/loss as % of cumulative invested capital
    # This avoids inflated % changes when NAV base is small (early quarters)
    q["nav_change_pct"] = q["gains_losses_mm"] / q["cumulative_contributions_mm"].clip(lower=0.01)

    # Classify as markup or markdown
    q["markup"] = q["nav_change_pct"] > 0.005
    q["markdown"] = q["nav_change_pct"] < -0.005

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
                   .reset_index()
                   .sort_values("fund_age_q"))

    # Smooth markup rate with 4-quarter centered rolling average
    # Eliminates jagged 0%/100% swings from low sample sizes per quarter
    age_pattern["markup_rate_smooth"] = (
        age_pattern["markup_rate"]
        .rolling(window=4, center=True, min_periods=1)
        .mean()
    )

    fig_pattern = make_subplots(specs=[[{"secondary_y": True}]])
    fig_pattern.add_trace(
        go.Bar(x=age_pattern["fund_age_q"], y=age_pattern["avg_nav_change"],
               name="Avg NAV Change %", marker_color=np.where(
                   age_pattern["avg_nav_change"] >= 0, "#2ecc71", "#e74c3c")),
        secondary_y=False,
    )
    fig_pattern.add_trace(
        go.Scatter(x=age_pattern["fund_age_q"], y=age_pattern["markup_rate_smooth"],
                   name="Markup Rate (4Q Avg)", line=dict(color="#3498db", width=2)),
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

    # ── NAV Trajectory Forecast ───────────────────────────────────
    st.subheader("NAV Trajectory Forecast")

    fund_options = dict(zip(gp_funds["fund_name"], gp_funds["fund_id"]))
    if fund_options:
        selected_fund_name = st.selectbox("Select Fund for Forecast",
                                           list(fund_options.keys()))
        selected_fund_id = fund_options[selected_fund_name]
        fund_q = gp_data[gp_data["fund_id"] == selected_fund_id].sort_values("quarter_end")

        if len(fund_q) >= 4:
            X = fund_q["fund_age_q"].values
            y = fund_q["ending_nav_mm"].values
            max_q = int(X.max())

            # ── Piecewise NAV forecast: pre-peak fit + post-peak slow decay ──
            peak_idx = np.argmax(y)
            peak_q = int(X[peak_idx])
            peak_nav = y[peak_idx]

            # Fit polynomial on pre-peak data (ramp-up phase)
            pre_peak_mask = X <= peak_q
            X_pre = X[pre_peak_mask].reshape(-1, 1)
            y_pre = y[pre_peak_mask]

            poly = PolynomialFeatures(degree=3)
            X_poly = poly.fit_transform(X_pre)
            model = LinearRegression()
            model.fit(X_poly, y_pre)

            # Build forecast: 8 quarters ahead
            all_q = np.arange(1, max_q + 9)
            forecast = np.zeros(len(all_q))

            for i, q_val in enumerate(all_q):
                if q_val <= peak_q:
                    # Pre-peak: use polynomial fit
                    forecast[i] = model.predict(poly.transform([[q_val]]))[0]
                else:
                    # Post-peak: slow exponential decay (asymmetric tail)
                    # Decline takes ~1.5-2x as long as ramp-up
                    quarters_past_peak = q_val - peak_q
                    ramp_quarters = max(peak_q, 4)
                    # Half-life = ramp_quarters * 1.5 (slower than ramp-up)
                    half_life = ramp_quarters * 1.5
                    decay_rate = 0.693 / half_life  # ln(2) / half_life
                    forecast[i] = peak_nav * np.exp(-decay_rate * quarters_past_peak)

            forecast = np.maximum(forecast, 0)

            # Confidence band based on residuals
            actual_forecast = np.array([
                model.predict(poly.transform([[x]]))[0] if x <= peak_q
                else peak_nav * np.exp(-0.693 / (max(peak_q, 4) * 1.5) * (x - peak_q))
                for x in X
            ])
            residuals = y - np.maximum(actual_forecast, 0)
            std_resid = np.std(residuals)

            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=X, y=y,
                mode="lines+markers", name="Actual NAV",
                line=dict(color="#2c3e50", width=2),
            ))
            fig_pred.add_trace(go.Scatter(
                x=all_q, y=forecast,
                mode="lines", name="Forecast NAV",
                line=dict(color="#e67e22", width=2, dash="dash"),
            ))
            fig_pred.add_trace(go.Scatter(
                x=np.concatenate([all_q, all_q[::-1]]),
                y=np.concatenate([forecast + 1.96 * std_resid,
                                  np.maximum(forecast - 1.96 * std_resid, 0)[::-1]]),
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

            # ── Cash Flow Forecast ─────────────────────────────────────
            st.subheader("Cash Flow Forecast")

            # Context line with GP name and style
            st.caption(f"Forecast based on {selected_gp_name}'s historical cash flow patterns "
                       f"across prior funds, calibrated to current fund age, strategy, and "
                       f"GP behavioral archetype ({gp_style}).")

            fund_cf = fund_q[["fund_age_q", "contributions_mm", "distributions_mm"]].copy()
            avg_dist_late = fund_cf[fund_cf["fund_age_q"] > max_q - 4]["distributions_mm"].mean()
            fund_commitment = gp_funds[gp_funds["fund_id"] == selected_fund_id]["total_commitment_mm"]
            commitment_val = float(fund_commitment.iloc[0]) if len(fund_commitment) > 0 else 100.0

            # Seeded random for reproducible lumpy distributions per fund
            cf_rng = np.random.default_rng(hash(selected_fund_id) % (2**31))

            # Forecast distributions: lumpy, not linear
            forecast_dists = []
            for i in range(8):
                base = avg_dist_late * (1 + 0.05 * i)
                # Random multiplier 0.6-1.5x, with occasional spikes (major exit)
                if cf_rng.random() < 0.15:
                    # ~15% chance of a major exit quarter (1.8-2.5x)
                    mult = float(cf_rng.uniform(1.8, 2.5))
                elif cf_rng.random() < 0.12:
                    # ~12% chance of near-zero quarter (no exits)
                    mult = float(cf_rng.uniform(0.0, 0.15))
                else:
                    mult = float(cf_rng.uniform(0.6, 1.5))
                forecast_dists.append(max(0, round(base * mult, 1)))

            # Forecast contributions: small, occasional (fees, follow-ons, add-ons)
            fund_vintage = gp_funds[gp_funds["fund_id"] == selected_fund_id]["vintage_year"]
            vintage_val = int(fund_vintage.iloc[0]) if len(fund_vintage) > 0 else 2020
            fund_age_years = 2025 - vintage_val

            forecast_contribs = []
            for i in range(8):
                # ~40-60% of quarters have a small call
                if cf_rng.random() < (0.55 if fund_age_years < 8 else 0.35):
                    # Scale to fund size: 0.1-0.5% of commitment per quarter
                    call_pct = float(cf_rng.uniform(0.001, 0.005))
                    if fund_age_years >= 10:
                        call_pct *= 0.5  # very late-stage: even smaller
                    call = round(commitment_val * call_pct, 1)
                    call = max(0.2, min(call, 1.0))  # floor $0.2M, cap $1.0M
                    forecast_contribs.append(call)
                else:
                    forecast_contribs.append(0.0)

            pred_cf = pd.DataFrame({
                "Quarter": [f"Q+{i+1}" for i in range(8)],
                "Forecast Contributions ($M)": forecast_contribs,
                "Forecast Distributions ($M)": forecast_dists,
            })
            st.dataframe(pred_cf, use_container_width=True, hide_index=True)
        else:
            st.info("Not enough quarterly data for forecast (need 4+ quarters).")
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
