"""
Module 5: Data Validation & QA Pipeline
Automated checks: NAV continuity, capital call reconciliation, outlier detection,
negative value flags, fee reasonableness.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def _run_validation(quarterly: pd.DataFrame, funds: pd.DataFrame) -> pd.DataFrame:
    """Run all QA checks and return a DataFrame of flagged issues."""
    df = quarterly.merge(funds[["fund_id", "fund_name"]], on="fund_id", how="left")
    df = df.sort_values(["fund_id", "quarter_end"]).reset_index(drop=True)

    issues = []

    # ── Check 1: NAV Continuity ────────────────────────────────────
    # Ending NAV of quarter N should match beginning NAV of quarter N+1
    for fund_id, group in df.groupby("fund_id"):
        group = group.sort_values("quarter_end").reset_index(drop=True)
        for i in range(1, len(group)):
            prev_end = group.loc[i - 1, "ending_nav_mm"]
            curr_begin = group.loc[i, "beginning_nav_mm"]
            if abs(prev_end - curr_begin) > 0.5:  # >$0.5M tolerance
                issues.append({
                    "fund_id": fund_id,
                    "fund_name": group.loc[i, "fund_name"],
                    "quarter": group.loc[i, "quarter_end"],
                    "check": "NAV Continuity",
                    "severity": "High" if abs(prev_end - curr_begin) > 5 else "Medium",
                    "detail": f"Previous ending NAV: ${prev_end:,.1f}M, "
                              f"Current beginning NAV: ${curr_begin:,.1f}M, "
                              f"Gap: ${prev_end - curr_begin:+,.1f}M",
                    "value": abs(prev_end - curr_begin),
                })

    # ── Check 2: Negative NAV ──────────────────────────────────────
    neg_nav = df[df["ending_nav_mm"] < 0]
    for _, row in neg_nav.iterrows():
        issues.append({
            "fund_id": row["fund_id"],
            "fund_name": row["fund_name"],
            "quarter": row["quarter_end"],
            "check": "Negative NAV",
            "severity": "Critical",
            "detail": f"Ending NAV is negative: ${row['ending_nav_mm']:,.1f}M",
            "value": abs(row["ending_nav_mm"]),
        })

    # ── Check 3: NAV Reconciliation ────────────────────────────────
    # beginning_nav + contributions - distributions - fees + gains = ending_nav
    df["recon_nav"] = (df["beginning_nav_mm"] + df["contributions_mm"]
                       - df["distributions_mm"] - df["mgmt_fees_mm"]
                       + df["gains_losses_mm"])
    df["recon_diff"] = abs(df["recon_nav"] - df["ending_nav_mm"])
    recon_issues = df[df["recon_diff"] > 1.0]
    for _, row in recon_issues.iterrows():
        issues.append({
            "fund_id": row["fund_id"],
            "fund_name": row["fund_name"],
            "quarter": row["quarter_end"],
            "check": "NAV Reconciliation",
            "severity": "High" if row["recon_diff"] > 5 else "Medium",
            "detail": f"Computed NAV: ${row['recon_nav']:,.1f}M vs "
                      f"Reported NAV: ${row['ending_nav_mm']:,.1f}M, "
                      f"Diff: ${row['recon_diff']:,.1f}M",
            "value": row["recon_diff"],
        })

    # ── Check 4: Outlier Gains/Losses ──────────────────────────────
    for fund_id, group in df.groupby("fund_id"):
        gains = group["gains_losses_mm"]
        if len(gains) < 4:
            continue
        mean_g = gains.mean()
        std_g = gains.std()
        if std_g < 0.01:
            continue
        outliers = group[abs(gains - mean_g) > 3 * std_g]
        for _, row in outliers.iterrows():
            issues.append({
                "fund_id": fund_id,
                "fund_name": row["fund_name"],
                "quarter": row["quarter_end"],
                "check": "Outlier Gain/Loss",
                "severity": "Medium",
                "detail": f"Gain/Loss: ${row['gains_losses_mm']:,.1f}M "
                          f"(Mean: ${mean_g:,.1f}M, Std: ${std_g:,.1f}M, "
                          f"Z-score: {(row['gains_losses_mm'] - mean_g)/std_g:+.1f})",
                "value": abs(row["gains_losses_mm"] - mean_g),
            })

    # ── Check 5: Fee Reasonableness ────────────────────────────────
    # Quarterly fee > 2% of NAV is suspicious
    fee_check = df[df["ending_nav_mm"] > 0].copy()
    fee_check["fee_pct"] = fee_check["mgmt_fees_mm"] / fee_check["ending_nav_mm"]
    fee_issues = fee_check[fee_check["fee_pct"] > 0.02]
    for _, row in fee_issues.iterrows():
        issues.append({
            "fund_id": row["fund_id"],
            "fund_name": row["fund_name"],
            "quarter": row["quarter_end"],
            "check": "Fee Spike",
            "severity": "Medium",
            "detail": f"Quarterly fee: ${row['mgmt_fees_mm']:,.1f}M = "
                      f"{row['fee_pct']:.1%} of NAV (threshold: 2%)",
            "value": row["mgmt_fees_mm"],
        })

    # ── Check 6: TVPI / DPI Consistency ────────────────────────────
    tvpi_check = df[(df["tvpi"] < 0) | (df["dpi"] < 0) | (df["dpi"] > df["tvpi"] + 0.01)]
    for _, row in tvpi_check.iterrows():
        issues.append({
            "fund_id": row["fund_id"],
            "fund_name": row["fund_name"],
            "quarter": row["quarter_end"],
            "check": "Multiple Consistency",
            "severity": "High",
            "detail": f"TVPI: {row['tvpi']:.3f}x, DPI: {row['dpi']:.3f}x, "
                      f"RVPI: {row['rvpi']:.3f}x — DPI should not exceed TVPI",
            "value": max(0, row["dpi"] - row["tvpi"]),
        })

    return pd.DataFrame(issues) if issues else pd.DataFrame(
        columns=["fund_id", "fund_name", "quarter", "check", "severity", "detail", "value"])


def render(quarterly_dirty: pd.DataFrame, funds: pd.DataFrame, gps: pd.DataFrame):
    st.header("Data Validation & QA Pipeline")
    st.caption("Automated quality checks on fund financial data. "
               "Issues are flagged with severity levels and actionable detail.")

    with st.expander("ℹ️ What checks are performed"):
        st.markdown("""
Six automated checks run on every quarterly record:

1. **NAV Continuity** — does the ending NAV of quarter N match the beginning NAV of quarter N+1?
2. **NAV Reconciliation** — does beginning NAV + contributions - fees + gains - distributions = ending NAV?
3. **Negative NAV** — is any reported NAV below zero? (rare and severe)
4. **Outlier Gain/Loss** — are any quarterly gains or losses more than 3 standard deviations from the fund's historical average?
5. **Fee Spike** — are management fees in any quarter abnormally high relative to NAV (>2% in a single quarter)?
6. **TVPI/DPI Consistency** — does DPI exceed TVPI? (impossible if data is correct, since TVPI = DPI + RVPI)
""")

    with st.expander("ℹ️ How severity is assigned"):
        st.markdown("""
- **Critical** — requires immediate attention, likely a data error (e.g. negative NAV)
- **High** — significant anomaly that needs investigation (e.g. large NAV reconciliation break, >$5M)
- **Medium** — minor inconsistency worth flagging (e.g. slightly elevated fee, moderate outlier)
""")

    with st.expander("ℹ️ How the data quality score works"):
        st.markdown("""
Starts at 100 and deducts points based on issue count weighted by severity, normalised by total records.
Critical issues penalise 5x, High 2x, Medium 1x. Formula: `100 - ((Critical×5 + High×2 + Medium×1) / Total Records × 100)`.
Score bands: **Good** (85+), **Acceptable** (65-84), **Needs Improvement** (40-64), **Critical** (<40).
""")

    issues_df = _run_validation(quarterly_dirty, funds)

    # ── Summary KPIs ───────────────────────────────────────────────
    n_total = len(quarterly_dirty)
    n_issues = len(issues_df)
    n_critical = len(issues_df[issues_df["severity"] == "Critical"]) if n_issues > 0 else 0
    n_high = len(issues_df[issues_df["severity"] == "High"]) if n_issues > 0 else 0
    n_medium = len(issues_df[issues_df["severity"] == "Medium"]) if n_issues > 0 else 0
    n_funds_affected = issues_df["fund_id"].nunique() if n_issues > 0 else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Records", f"{n_total:,}")
    k2.metric("Issues Found", f"{n_issues}", delta=None)
    k3.metric("Critical", f"{n_critical}", delta="Immediate action" if n_critical > 0 else None,
              delta_color="inverse")
    k4.metric("High", f"{n_high}")
    k5.metric("Funds Affected", f"{n_funds_affected}")

    if n_issues == 0:
        st.success("All validation checks passed. No issues detected.")
        return

    st.divider()

    # ── Issue Breakdown ────────────────────────────────────────────
    left, right = st.columns(2)

    with left:
        st.subheader("Issues by Check Type")
        check_counts = issues_df["check"].value_counts().reset_index()
        check_counts.columns = ["Check", "Count"]
        fig_check = px.bar(check_counts, x="Count", y="Check", orientation="h",
                           color="Count", color_continuous_scale="Reds")
        fig_check.update_layout(height=350, coloraxis_showscale=False,
                                margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_check, use_container_width=True)

    with right:
        st.subheader("Issues by Severity")
        sev_counts = issues_df["severity"].value_counts().reset_index()
        sev_counts.columns = ["Severity", "Count"]
        color_map = {"Critical": "#e74c3c", "High": "#e67e22", "Medium": "#f39c12"}
        fig_sev = px.pie(sev_counts, values="Count", names="Severity",
                         color="Severity", color_discrete_map=color_map, hole=0.4)
        fig_sev.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_sev, use_container_width=True)

    st.divider()

    # ── Issues by Fund ─────────────────────────────────────────────
    st.subheader("Issues by Fund")
    fund_issues = (issues_df.groupby(["fund_id", "fund_name"])
                   .agg(n_issues=("check", "count"),
                        critical=("severity", lambda x: (x == "Critical").sum()),
                        high=("severity", lambda x: (x == "High").sum()))
                   .reset_index()
                   .sort_values("n_issues", ascending=False))
    fig_fund = px.bar(fund_issues.head(15), x="fund_name", y="n_issues",
                      color="critical",
                      labels={"n_issues": "# Issues", "fund_name": "Fund",
                              "critical": "Critical Issues"},
                      color_continuous_scale="Reds")
    fig_fund.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_fund, use_container_width=True)

    st.divider()

    # ── Issues Timeline ────────────────────────────────────────────
    st.subheader("Issues Over Time")
    timeline = issues_df.copy()
    timeline["quarter_dt"] = pd.to_datetime(timeline["quarter"])
    # Group by actual datetime (not string) so sorting is inherently chronological
    timeline_agg = (timeline.groupby(["quarter_dt", "check"])
                    .size().reset_index(name="count")
                    .sort_values("quarter_dt"))
    fig_time = px.bar(timeline_agg, x="quarter_dt", y="count", color="check",
                      barmode="stack", labels={"quarter_dt": "Quarter",
                                                "count": "Count", "check": "Check"})
    fig_time.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
    fig_time.update_xaxes(tickangle=45, dtick="M6",  # tick every 6 months
                          tickformat="%YQ%q")
    st.plotly_chart(fig_time, use_container_width=True)

    st.divider()

    # ── Detailed Issues Table ──────────────────────────────────────
    st.subheader("Detailed Issue Log")

    # Filters
    f1, f2, f3 = st.columns(3)
    with f1:
        sev_filter = st.multiselect("Severity", ["Critical", "High", "Medium"],
                                     default=["Critical", "High", "Medium"],
                                     key="qa_sev")
    with f2:
        check_filter = st.multiselect("Check Type", issues_df["check"].unique().tolist(),
                                       default=issues_df["check"].unique().tolist(),
                                       key="qa_check")
    with f3:
        fund_filter = st.multiselect("Fund", issues_df["fund_name"].unique().tolist(),
                                      default=issues_df["fund_name"].unique().tolist(),
                                      key="qa_fund")

    filtered = issues_df[
        (issues_df["severity"].isin(sev_filter)) &
        (issues_df["check"].isin(check_filter)) &
        (issues_df["fund_name"].isin(fund_filter))
    ]

    display = filtered[["fund_name", "quarter", "check", "severity", "detail"]].copy()
    display.columns = ["Fund", "Quarter", "Check Type", "Severity", "Detail"]

    def color_severity(val):
        colors = {"Critical": "background-color: #e74c3c; color: white",
                  "High": "background-color: #e67e22; color: white",
                  "Medium": "background-color: #f39c12"}
        return colors.get(val, "")

    st.dataframe(
        display.style.map(color_severity, subset=["Severity"]),
        use_container_width=True, height=500,
    )

    # ── Data Quality Score ─────────────────────────────────────────
    st.divider()
    st.subheader("Overall Data Quality Score")

    # Weighted penalty normalised by total records:
    # Critical 5x, High 2x, Medium 1x
    weighted_issues = n_critical * 5 + n_high * 2 + n_medium * 1
    penalty = (weighted_issues / max(n_total, 1)) * 100
    quality_score = max(0, min(100, round(100 - penalty)))

    # Score band label
    if quality_score >= 85:
        score_label = "Good"
    elif quality_score >= 65:
        score_label = "Acceptable"
    elif quality_score >= 40:
        score_label = "Needs Improvement"
    else:
        score_label = "Critical"

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=quality_score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": f"Data Quality Score — {score_label}"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#2ecc71" if quality_score >= 80 else
                    "#f39c12" if quality_score >= 60 else "#e74c3c"},
            "steps": [
                {"range": [0, 40], "color": "#fde8e8"},
                {"range": [40, 65], "color": "#fef3e0"},
                {"range": [65, 85], "color": "#fef9e0"},
                {"range": [85, 100], "color": "#e8f8e8"},
            ],
            "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 85},
        },
    ))
    fig_gauge.update_layout(height=350, margin=dict(l=40, r=40, t=60, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)
