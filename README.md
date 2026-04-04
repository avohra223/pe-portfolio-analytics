# PE Portfolio Analytics Platform

A comprehensive private equity portfolio analytics platform built with Python and Streamlit. Covers fund performance monitoring, concentration risk mapping, GP behavior analysis, secondary pricing, data validation, and macro stress testing across a synthetic dataset of 32 funds, 200 portfolio companies, and 10 vintages (2014-2023).

**Live Dashboard:** [avohra223-pe-portfolio-analytics.streamlit.app](https://avohra223-pe-portfolio-analytics.streamlit.app/)

---

## Modules

### 1. Portfolio Monitoring Dashboard
Fund performance tracking with IRR, TVPI, DPI, and RVPI metrics. Exposure breakdowns by strategy, geography, and vintage year. Aggregate cash flow timelines and commitment tracking (funded vs unfunded).

### 2. Hidden Concentration Risk Mapper
Entity resolution across funds to identify portfolio companies held in multiple funds. Network graph visualization of fund-level overlaps, sector/geography concentration heatmaps, and correlated exposure analysis. Identifies 86 companies appearing in 2+ funds.

### 3. GP Behavior Prediction Engine
Analyzes historical GP markup/markdown patterns, distribution timing, and fund life management. Polynomial regression-based NAV trajectory prediction with confidence intervals. Compares GP behavioral archetypes (aggressive, conservative, balanced).

### 4. Secondary Pricing Model
Estimates secondary market bid price as % of NAV using gradient boosting. Features include fund age, strategy, DPI, unfunded ratio, and GP track record. Interactive pricing simulator for custom scenario analysis. Model validated with 5-fold cross-validation.

### 5. Data Validation & QA Pipeline
Six automated checks: NAV continuity, NAV reconciliation, negative NAV detection, outlier gain/loss flagging, fee reasonableness, and TVPI/DPI consistency. Severity-based issue classification (Critical/High/Medium) with a portfolio-level data quality score.

### 6. Macro Stress Test Simulator
Models how macro shocks propagate from portfolio companies to fund and portfolio-level NAV. Six preset scenarios (rate hike, mild/deep recession, stagflation, tech crash, energy crisis) with sector-specific sensitivity betas. Adjustable severity multiplier and custom scenario parameters.

---

## Data Model

All analytics run on a synthetic dataset designed with realistic PE characteristics:

- **12 GPs** with behavioral archetypes (aggressive/conservative/balanced)
- **32 funds** across 6 strategies (Buyout, Growth Equity, VC, Distressed, Real Estate, Infrastructure)
- **200 portfolio companies** across 8 sectors, with deliberate multi-fund overlaps for concentration risk
- **1,000+ quarterly records** with J-curve NAV paths calibrated by vintage
- **IRR computed from actual cash flow series** via XIRR/Brent's method (not approximated)
- **NAV identity enforced:** ending_nav = beginning_nav + contributions - fees + gains - distributions
- **TVPI = (Distributions + NAV) / Funded** — internally consistent for every fund, every quarter

### Vintage Calibration

| Vintage | Avg IRR | Avg TVPI | Avg DPI | Stage |
|---------|---------|----------|---------|-------|
| 2014-2016 | 12-14% | 2.1-2.2x | 1.3-1.7x | Mature |
| 2017-2018 | 11-12% | 1.8-1.9x | 0.9-1.0x | Maturing |
| 2019-2020 | 9-10% | 1.4-1.5x | 0.3-0.5x | Mid-life |
| 2021-2022 | 3-7% | 1.1-1.2x | 0.0-0.1x | Early |
| 2023 | -7% | 0.9x | 0.0x | J-curve |

---

## Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| Data Processing | pandas, NumPy |
| Visualization | Plotly |
| ML Models | scikit-learn (Gradient Boosting), SciPy (XIRR) |
| Network Analysis | NetworkX |
| Deployment | Streamlit Community Cloud |

---

## Run Locally

```bash
git clone https://github.com/avohra223/pe-portfolio-analytics.git
cd pe-portfolio-analytics
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
streamlit run app.py
```

---

## Project Structure

```
pe-portfolio-analytics/
├── app.py                    # Main Streamlit app with sidebar navigation
├── data_generator.py         # Synthetic PE data engine (32 funds, 200 companies)
├── requirements.txt
└── modules/
    ├── dashboard.py          # Portfolio Monitoring Dashboard
    ├── concentration.py      # Hidden Concentration Risk Mapper
    ├── gp_behavior.py        # GP Behavior Prediction Engine
    ├── pricing.py            # Secondary Pricing Model
    ├── validation.py         # Data Validation & QA Pipeline
    └── stress_test.py        # Macro Stress Test Simulator
```
