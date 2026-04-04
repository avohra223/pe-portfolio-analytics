"""
PE Portfolio Analytics Platform
==========================================
A comprehensive private equity analytics platform with 6 modules:
1. Portfolio Monitoring Dashboard
2. Hidden Concentration Risk Mapper
3. GP Behavior Prediction Engine
4. Secondary Pricing Model
5. Data Validation & QA Pipeline
6. Macro Stress Test Simulator
"""

import streamlit as st

st.set_page_config(
    page_title="PE Portfolio Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global */
    .stApp { background-color: #0e1117; }
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #21262d;
    }
    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #161b22, #1c2333);
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 16px;
    }
    [data-testid="stMetricValue"] { font-size: 1.4rem; }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #161b22;
        border-radius: 8px 8px 0 0;
        border: 1px solid #21262d;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1c2333;
        border-bottom: 2px solid #58a6ff;
    }
    /* Dividers */
    hr { border-color: #21262d !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.title("PE Portfolio Analytics")
    st.caption("Private Equity Portfolio Monitoring & Analytics")
    st.divider()

    module = st.radio(
        "Module",
        [
            "Portfolio Dashboard",
            "Concentration Risk",
            "GP Behavior Engine",
            "Secondary Pricing",
            "Data Validation & QA",
            "Macro Stress Test",
        ],
        index=0,
    )

    st.divider()
    st.caption("**Dataset:** 30 funds | 200 companies | 10 vintages")
    st.caption("Synthetic data for demonstration purposes")
    st.caption("Synthetic data for demonstration")


# ── Data Loading ───────────────────────────────────────────────────
@st.cache_data(show_spinner="Generating synthetic PE data...")
def load_data():
    from data_generator import generate_all_data
    return generate_all_data()


data = load_data()

gps = data["gps"]
funds = data["funds"]
companies = data["companies"]
holdings = data["holdings"]
quarterly = data["quarterly"]
quarterly_dirty = data["quarterly_dirty"]
cash_flows = data["cash_flows"]


# ── Module Routing ─────────────────────────────────────────────────
if module == "Portfolio Dashboard":
    from modules.dashboard import render
    render(funds, quarterly, cash_flows, gps)

elif module == "Concentration Risk":
    from modules.concentration import render
    render(funds, companies, holdings, gps)

elif module == "GP Behavior Engine":
    from modules.gp_behavior import render
    render(funds, quarterly, gps)

elif module == "Secondary Pricing":
    from modules.pricing import render
    render(funds, quarterly, gps)

elif module == "Data Validation & QA":
    from modules.validation import render
    render(quarterly_dirty, funds, gps)

elif module == "Macro Stress Test":
    from modules.stress_test import render
    render(funds, quarterly, companies, holdings, gps)
