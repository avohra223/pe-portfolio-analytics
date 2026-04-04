"""
Module 2: Hidden Concentration Risk Mapper
Entity resolution across funds, showing which portfolio companies appear in
multiple funds, sector/geography overlap, correlated exposure visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx


def render(funds: pd.DataFrame, companies: pd.DataFrame, holdings: pd.DataFrame,
           gps: pd.DataFrame):
    st.header("Hidden Concentration Risk Mapper")
    st.caption("Identifies portfolio companies held across multiple funds, "
               "revealing hidden sector, geographic, and entity-level concentration.")

    # ── Enriched holdings ──────────────────────────────────────────
    h = (holdings
         .merge(companies, on="company_id", how="left")
         .merge(funds[["fund_id", "fund_name", "strategy", "geography", "gp_id"]], on="fund_id", how="left")
         .merge(gps[["gp_id", "gp_name"]], on="gp_id", how="left"))

    # ── Overlap detection ──────────────────────────────────────────
    fund_counts = h.groupby("company_id")["fund_id"].nunique().reset_index()
    fund_counts.columns = ["company_id", "n_funds"]
    overlap_cos = fund_counts[fund_counts["n_funds"] >= 2]["company_id"]
    overlap_holdings = h[h["company_id"].isin(overlap_cos)]

    n_overlap = len(overlap_cos)
    n_total = companies.shape[0]

    # ── KPI Row ────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Portfolio Companies", f"{n_total}")
    k2.metric("Multi-Fund Overlaps", f"{n_overlap}")
    k3.metric("Overlap Rate", f"{n_overlap/n_total:.1%}")
    max_overlap = fund_counts["n_funds"].max() if len(fund_counts) > 0 else 0
    k4.metric("Max Fund Overlap", f"{max_overlap} funds")

    st.divider()

    # ── Entity Overlap Network Graph ───────────────────────────────
    st.subheader("Entity Overlap Network")
    st.caption("Funds are connected when they share a portfolio company. "
               "Thicker edges = more shared companies.")

    # Build adjacency: fund-to-fund via shared companies
    overlap_detail = overlap_holdings.groupby("company_id")["fund_id"].apply(list).reset_index()
    edge_weights = {}
    for _, row in overlap_detail.iterrows():
        fids = sorted(set(row["fund_id"]))
        for i in range(len(fids)):
            for j in range(i + 1, len(fids)):
                key = (fids[i], fids[j])
                edge_weights[key] = edge_weights.get(key, 0) + 1

    G = nx.Graph()
    fund_name_map = dict(zip(funds["fund_id"], funds["fund_name"]))
    for fid in funds["fund_id"]:
        G.add_node(fid, label=fund_name_map.get(fid, fid))
    for (f1, f2), w in edge_weights.items():
        G.add_edge(f1, f2, weight=w)

    pos = nx.spring_layout(G, seed=42, k=2.5)

    # Plotly network visualization
    edge_x, edge_y = [], []
    edge_widths = []
    for (f1, f2), w in edge_weights.items():
        x0, y0 = pos[f1]
        x1, y1 = pos[f2]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_widths.append(w)

    avg_width = np.mean(list(edge_weights.values())) if edge_weights else 1

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=max(1, avg_width * 0.8), color="#b0bec5"),
        hoverinfo="none",
    )

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_text = [fund_name_map.get(n, n) for n in G.nodes()]
    node_degree = [G.degree(n, weight="weight") for n in G.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(size=[max(12, d * 2) for d in node_degree],
                    color=node_degree, colorscale="YlOrRd",
                    colorbar=dict(title="Overlap<br>Weight"),
                    line=dict(width=1, color="#333")),
        text=node_text, textposition="top center", textfont=dict(size=9),
        hovertext=[f"{t}<br>Overlap weight: {d}" for t, d in zip(node_text, node_degree)],
        hoverinfo="text",
    )

    fig_net = go.Figure(data=[edge_trace, node_trace])
    fig_net.update_layout(
        showlegend=False, height=550,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig_net, use_container_width=True)

    st.divider()

    # ── Top Overlapping Companies Table ────────────────────────────
    st.subheader("Most Concentrated Portfolio Companies")
    top_overlap = (overlap_holdings
                   .groupby(["company_id", "company_name", "sector", "region"])
                   .agg(n_funds=("fund_id", "nunique"),
                        total_cost=("initial_cost_mm", "sum"),
                        funds_list=("fund_name", lambda x: ", ".join(sorted(set(x)))))
                   .reset_index()
                   .sort_values("n_funds", ascending=False)
                   .head(20))
    top_overlap.columns = ["ID", "Company", "Sector", "Region", "# Funds",
                           "Total Cost ($M)", "Funds Holding"]
    st.dataframe(
        top_overlap[["Company", "Sector", "Region", "# Funds", "Total Cost ($M)",
                      "Funds Holding"]].style.format({"Total Cost ($M)": "{:,.1f}"}),
        use_container_width=True, height=500,
    )

    st.divider()

    # ── Sector Concentration ───────────────────────────────────────
    left, right = st.columns(2)

    with left:
        st.subheader("Sector Exposure (by Cost)")
        sector_exp = h.groupby("sector")["initial_cost_mm"].sum().reset_index()
        sector_exp = sector_exp.sort_values("initial_cost_mm", ascending=True)
        fig_sec = px.bar(sector_exp, y="sector", x="initial_cost_mm", orientation="h",
                         labels={"initial_cost_mm": "Total Cost ($M)", "sector": ""},
                         color="initial_cost_mm", color_continuous_scale="Blues")
        fig_sec.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20),
                              coloraxis_showscale=False)
        st.plotly_chart(fig_sec, use_container_width=True)

    with right:
        st.subheader("Geographic Exposure (by Cost)")
        geo_exp = h.groupby("region")["initial_cost_mm"].sum().reset_index()
        fig_geo = px.pie(geo_exp, values="initial_cost_mm", names="region", hole=0.4)
        fig_geo.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_geo, use_container_width=True)

    st.divider()

    # ── Sector × Strategy Heatmap ──────────────────────────────────
    st.subheader("Sector x Strategy Concentration Heatmap")
    cross = h.groupby(["sector", "strategy"])["initial_cost_mm"].sum().reset_index()
    pivot = cross.pivot_table(index="sector", columns="strategy",
                               values="initial_cost_mm", fill_value=0)
    fig_heat = px.imshow(
        pivot, text_auto=".0f", aspect="auto",
        labels=dict(x="Strategy", y="Sector", color="Cost ($M)"),
        color_continuous_scale="YlOrRd",
    )
    fig_heat.update_layout(height=450, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Sector × Geography Heatmap ─────────────────────────────────
    st.subheader("Sector x Geography Concentration Heatmap")
    cross2 = h.groupby(["sector", "region"])["initial_cost_mm"].sum().reset_index()
    pivot2 = cross2.pivot_table(index="sector", columns="region",
                                 values="initial_cost_mm", fill_value=0)
    fig_heat2 = px.imshow(
        pivot2, text_auto=".0f", aspect="auto",
        labels=dict(x="Geography", y="Sector", color="Cost ($M)"),
        color_continuous_scale="Viridis",
    )
    fig_heat2.update_layout(height=450, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_heat2, use_container_width=True)
