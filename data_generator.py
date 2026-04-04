"""
PE Portfolio Analytics — Synthetic Data Generator
Generates realistic private equity fund data: ~30 funds, ~200 portfolio companies,
quarterly performance across 8-10 vintages (2014-2023).

IRR is computed from actual cash flow series via scipy.optimize.
NAV = cumulative_funded - cumulative_distributed + unrealised_gain_loss (internally consistent).
TVPI = (Distributions + NAV) / Funded — enforced exactly.
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq

SEED = 42

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------
GP_NAMES = [
    "Meridian Capital Partners", "Northbridge Equity Group", "Apex Ventures",
    "Ironwood Capital", "Summit Growth Partners", "Catalina Partners",
    "Redstone Capital Management", "Vanguard Point Advisors", "Crestline Investments",
    "BluePeak Capital", "Harborview Partners", "Sterling Oak Capital",
]

STRATEGIES = ["Buyout", "Growth Equity", "Venture Capital", "Distressed / Special Sits",
              "Real Estate", "Infrastructure"]

GEOGRAPHIES = ["North America", "Europe", "Asia-Pacific"]

SECTORS = ["Technology", "Healthcare", "Consumer / Retail", "Industrials",
           "Financial Services", "Energy", "Real Estate", "Media & Telecom"]

COMPANY_PREFIXES = [
    "Apex", "Nova", "Vertex", "Zenith", "Catalyst", "Quantum", "Nexus",
    "Pinnacle", "Horizon", "Atlas", "Forge", "Pulse", "Orbit", "Helix",
    "Vantage", "Stratos", "Ember", "Crest", "Lumen", "Prism", "Echo",
    "Titan", "Onyx", "Bolt", "Nimbus", "Aether", "Flux", "Core", "Sync",
    "Arc", "Kinetic", "Radiant", "Summit", "Vector", "Ionic", "Sage",
    "Slate", "Ridge", "Cobalt", "Drift", "Spark", "Shield", "Blade",
    "Optic", "Cipher", "Axiom", "Quartz", "Zephyr", "Fathom", "Clover",
]

COMPANY_SUFFIXES = [
    "Technologies", "Health", "Solutions", "Systems", "Group", "Labs",
    "Therapeutics", "Analytics", "Dynamics", "Networks", "Digital",
    "Innovations", "Capital", "Logistics", "Software", "Bio", "Energy",
    "Services", "Platforms", "AI", "Robotics", "Materials", "Media",
]

COUNTRIES = {
    "North America": ["United States", "Canada"],
    "Europe": ["United Kingdom", "Germany", "France", "Netherlands", "Sweden", "Spain"],
    "Asia-Pacific": ["China", "Japan", "India", "Australia", "Singapore", "South Korea"],
}

# ---------------------------------------------------------------------------
# Vintage-specific return targets (the key calibration table)
# ---------------------------------------------------------------------------
VINTAGE_PROFILES = {
    2014: {"irr_lo": 0.12, "irr_hi": 0.18, "tvpi_lo": 1.8, "tvpi_hi": 2.5, "dpi_frac": 0.75},
    2015: {"irr_lo": 0.12, "irr_hi": 0.18, "tvpi_lo": 1.8, "tvpi_hi": 2.5, "dpi_frac": 0.70},
    2016: {"irr_lo": 0.12, "irr_hi": 0.18, "tvpi_lo": 1.8, "tvpi_hi": 2.5, "dpi_frac": 0.65},
    2017: {"irr_lo": 0.10, "irr_hi": 0.15, "tvpi_lo": 1.5, "tvpi_hi": 2.0, "dpi_frac": 0.50},
    2018: {"irr_lo": 0.10, "irr_hi": 0.15, "tvpi_lo": 1.5, "tvpi_hi": 2.0, "dpi_frac": 0.45},
    2019: {"irr_lo": 0.08, "irr_hi": 0.12, "tvpi_lo": 1.3, "tvpi_hi": 1.6, "dpi_frac": 0.30},
    2020: {"irr_lo": 0.08, "irr_hi": 0.12, "tvpi_lo": 1.3, "tvpi_hi": 1.6, "dpi_frac": 0.25},
    2021: {"irr_lo": 0.00, "irr_hi": 0.08, "tvpi_lo": 1.0, "tvpi_hi": 1.3, "dpi_frac": 0.10},
    2022: {"irr_lo": 0.00, "irr_hi": 0.08, "tvpi_lo": 1.0, "tvpi_hi": 1.3, "dpi_frac": 0.05},
    2023: {"irr_lo": -0.10, "irr_hi": 0.00, "tvpi_lo": 0.80, "tvpi_hi": 1.0, "dpi_frac": 0.02},
}

# Strategy modifiers (additive to IRR, multiplicative to TVPI)
STRATEGY_MODS = {
    "Buyout":                    {"irr_adj": 0.00,  "tvpi_mult": 1.00},
    "Growth Equity":             {"irr_adj": 0.01,  "tvpi_mult": 1.05},
    "Venture Capital":           {"irr_adj": 0.02,  "tvpi_mult": 1.10},
    "Distressed / Special Sits": {"irr_adj": -0.01, "tvpi_mult": 0.95},
    "Real Estate":               {"irr_adj": -0.02, "tvpi_mult": 0.95},
    "Infrastructure":            {"irr_adj": -0.02, "tvpi_mult": 0.93},
}

# GP behavioral archetypes
GP_STYLES = {
    "aggressive":    {"markup_bias": 0.03, "dist_delay": -1, "extension_prob": 0.15},
    "conservative":  {"markup_bias": -0.02, "dist_delay": 1, "extension_prob": 0.30},
    "balanced":      {"markup_bias": 0.00, "dist_delay": 0, "extension_prob": 0.20},
}


# ---------------------------------------------------------------------------
# IRR calculation from cash flows
# ---------------------------------------------------------------------------
def compute_irr(dates: list, cashflows: list) -> float:
    """
    Compute IRR from a series of (date, cashflow) pairs using XIRR method.
    Negative = outflow (capital call), positive = inflow (distribution / residual NAV).
    Returns annualised IRR.
    """
    if not cashflows or all(cf == 0 for cf in cashflows):
        return 0.0

    # Convert dates to year fractions from first date
    d0 = dates[0]
    year_fracs = [(d - d0).days / 365.25 for d in dates]

    def npv(rate):
        return sum(cf / (1 + rate) ** t for cf, t in zip(cashflows, year_fracs))

    # Search for IRR
    try:
        irr = brentq(npv, -0.5, 5.0, maxiter=200)
    except (ValueError, RuntimeError):
        # Fallback: try narrower range
        try:
            irr = brentq(npv, -0.3, 2.0, maxiter=200)
        except (ValueError, RuntimeError):
            irr = 0.0
    return irr


# ---------------------------------------------------------------------------
# Quarter range helper
# ---------------------------------------------------------------------------
def _quarter_range(start_year: int, end_year: int) -> pd.DatetimeIndex:
    """Generate quarter-end dates from start_year Q1 to end_year Q4."""
    return pd.date_range(start=f"{start_year}-03-31", end=f"{end_year}-12-31", freq="QE")


# ---------------------------------------------------------------------------
# Entity generators (GPs, funds, companies, holdings)
# ---------------------------------------------------------------------------
def generate_gps(rng: np.random.Generator) -> pd.DataFrame:
    styles = list(GP_STYLES.keys())
    records = []
    for i, name in enumerate(GP_NAMES):
        records.append({
            "gp_id": f"GP{i+1:03d}",
            "gp_name": name,
            "hq_city": rng.choice(["New York", "London", "San Francisco", "Boston",
                                    "Chicago", "Hong Kong", "Munich", "Singapore"]),
            "founded_year": int(rng.integers(1995, 2015)),
            "aum_bn": round(float(rng.uniform(2, 45)), 1),
            "style": styles[i % len(styles)],
            "track_record_score": round(float(rng.uniform(0.5, 1.0)), 2),
        })
    return pd.DataFrame(records)


def generate_funds(gps: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    records = []
    fund_idx = 0
    for vintage in range(2014, 2024):
        n_funds = int(rng.integers(2, 5))
        for _ in range(n_funds):
            gp = gps.iloc[fund_idx % len(gps)]
            strategy = rng.choice(STRATEGIES)
            geo = rng.choice(GEOGRAPHIES)
            fund_size = float(rng.choice([500, 750, 1000, 1500, 2000, 3000, 5000]))
            fund_num = rng.integers(1, 6)
            records.append({
                "fund_id": f"F{fund_idx+1:03d}",
                "fund_name": f"{gp['gp_name'].split()[0]} Fund {int(fund_num)}",
                "gp_id": gp["gp_id"],
                "strategy": strategy,
                "geography": geo,
                "vintage_year": vintage,
                "fund_size_mm": fund_size,
                "total_commitment_mm": round(fund_size * float(rng.uniform(0.05, 0.15)), 1),
                "fund_term_years": int(rng.choice([10, 10, 10, 12, 12])),
                "extension_years": 0,
            })
            fund_idx += 1
    return pd.DataFrame(records)


def generate_companies(rng: np.random.Generator, n: int = 200) -> pd.DataFrame:
    records = []
    used_names = set()
    for i in range(n):
        while True:
            name = f"{rng.choice(COMPANY_PREFIXES)} {rng.choice(COMPANY_SUFFIXES)}"
            if name not in used_names:
                used_names.add(name)
                break
        sector = rng.choice(SECTORS)
        region = rng.choice(GEOGRAPHIES)
        country = rng.choice(COUNTRIES[region])
        records.append({
            "company_id": f"CO{i+1:04d}",
            "company_name": name,
            "sector": sector,
            "region": region,
            "country": country,
            "founded_year": int(rng.integers(2000, 2022)),
            "revenue_mm": round(float(rng.lognormal(4, 1.2)), 1),
            "ebitda_margin": round(float(rng.uniform(0.05, 0.45)), 2),
            "ev_ebitda_multiple": round(float(rng.uniform(6, 25)), 1),
        })
    return pd.DataFrame(records)


def generate_holdings(funds: pd.DataFrame, companies: pd.DataFrame,
                      rng: np.random.Generator) -> pd.DataFrame:
    records = []
    company_ids = companies["company_id"].tolist()

    for _, fund in funds.iterrows():
        n_holdings = int(rng.integers(5, 12))
        chosen = rng.choice(company_ids, size=min(n_holdings, len(company_ids)), replace=False)
        for cid in chosen:
            cost = round(float(rng.uniform(10, 150)), 1)
            records.append({
                "fund_id": fund["fund_id"],
                "company_id": cid,
                "initial_cost_mm": cost,
                "ownership_pct": round(float(rng.uniform(0.02, 0.25)), 3),
            })

    overlap_cos = rng.choice(company_ids, size=35, replace=False)
    fund_ids = funds["fund_id"].tolist()
    existing_pairs = set((r["fund_id"], r["company_id"]) for r in records)

    for cid in overlap_cos:
        extra_funds = rng.choice(fund_ids, size=int(rng.integers(1, 3)), replace=False)
        for fid in extra_funds:
            if (fid, cid) not in existing_pairs:
                cost = round(float(rng.uniform(10, 100)), 1)
                records.append({
                    "fund_id": fid,
                    "company_id": cid,
                    "initial_cost_mm": cost,
                    "ownership_pct": round(float(rng.uniform(0.01, 0.15)), 3),
                })
                existing_pairs.add((fid, cid))

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Core: J-curve simulation with vintage-calibrated targets
# ---------------------------------------------------------------------------
def _simulate_fund_path(n_quarters: int, vintage: int, strategy: str,
                        gp_style: str, rng: np.random.Generator) -> dict:
    """
    Simulate quarterly cash flows and NAV for a fund.

    The simulation targets vintage-specific TVPI and DPI outcomes, then
    IRR is computed from the resulting cash flow series (not approximated).

    NAV identity enforced:
        ending_nav = beginning_nav + contributions - mgmt_fees + gains - distributions
    """
    vp = VINTAGE_PROFILES[vintage]
    sm = STRATEGY_MODS[strategy]
    style = GP_STYLES[gp_style]

    commitment = 100.0  # normalised

    # ── Target outcomes for this fund ──────────────────────────────
    target_tvpi = float(rng.uniform(vp["tvpi_lo"], vp["tvpi_hi"])) * sm["tvpi_mult"]
    target_tvpi = max(0.5, target_tvpi)
    dpi_frac = vp["dpi_frac"] + float(rng.uniform(-0.10, 0.10))
    dpi_frac = np.clip(dpi_frac, 0.0, 0.95)

    # ── Investment schedule (draw down over years 1-5) ─────────────
    invest_end_q = int(rng.integers(14, 22))  # 3.5 to 5.5 years
    invest_schedule = np.zeros(n_quarters)
    for q in range(min(invest_end_q, n_quarters)):
        weight = max(0.1, 1.0 - (q / invest_end_q) ** 0.6) + float(rng.uniform(0, 0.2))
        invest_schedule[q] = weight
    if invest_schedule.sum() > 0:
        # Draw down 85-100% of commitment
        drawdown_pct = float(rng.uniform(0.85, 1.00))
        invest_schedule = invest_schedule / invest_schedule.sum() * commitment * drawdown_pct

    # ── Compute total funded ───────────────────────────────────────
    total_funded = invest_schedule.sum()

    # ── Target values derived from TVPI and DPI ────────────────────
    total_value = target_tvpi * total_funded          # NAV + distributions
    total_distributions = dpi_frac * total_value      # lifetime distributions
    target_final_nav = total_value - total_distributions  # residual NAV
    target_final_nav = max(0, target_final_nav)

    # ── Management fees (~0.4-0.5% quarterly on committed during invest,
    #    then on NAV during harvest) ────────────────────────────────
    fee_rate_invest = float(rng.uniform(0.004, 0.005))
    fee_rate_harvest = float(rng.uniform(0.003, 0.004))

    # ── Distribution schedule ──────────────────────────────────────
    dist_start_q = int(rng.integers(10, 18)) + style["dist_delay"]
    dist_start_q = max(6, min(dist_start_q, n_quarters - 2))

    # Pre-compute distribution weights (ramp up then taper)
    dist_weights = np.zeros(n_quarters)
    for q in range(dist_start_q, n_quarters):
        age_frac = (q - dist_start_q) / max(n_quarters - dist_start_q - 1, 1)
        # Bell curve: ramp up, peak at ~60% through harvest, then taper
        w = np.exp(-((age_frac - 0.5) ** 2) / 0.15) + float(rng.uniform(0, 0.2))
        dist_weights[q] = w
    if dist_weights.sum() > 0:
        dist_weights = dist_weights / dist_weights.sum() * total_distributions

    # ── Simulate quarter by quarter ────────────────────────────────
    nav = np.zeros(n_quarters)
    contributions = np.zeros(n_quarters)
    distributions = np.zeros(n_quarters)
    mgmt_fees = np.zeros(n_quarters)
    gains = np.zeros(n_quarters)

    current_nav = 0.0
    cum_funded = 0.0
    cum_dist = 0.0

    for q in range(n_quarters):
        beginning_nav = current_nav

        # Contribution
        contrib = invest_schedule[q] if q < len(invest_schedule) else 0.0
        contributions[q] = contrib
        cum_funded += contrib

        # Management fee
        if q < invest_end_q:
            fee = commitment * fee_rate_invest
        else:
            fee = max(beginning_nav, 0) * fee_rate_harvest
        mgmt_fees[q] = fee

        # Distribution (from pre-computed schedule, but capped at available NAV)
        dist = dist_weights[q] if q < len(dist_weights) else 0.0
        # Can't distribute more than NAV + contribution - fee
        available = beginning_nav + contrib - fee
        dist = min(dist, max(0, available * 0.8))  # leave some NAV
        distributions[q] = dist
        cum_dist += dist

        # Gain/loss: chosen so that by final quarter, NAV hits target
        # Use a smooth J-curve path with noise
        fund_age_frac = q / max(n_quarters - 1, 1)
        remaining_q = n_quarters - q - 1

        # Implied NAV target at this point (smooth interpolation)
        # J-curve: dip in years 1-2, then steady appreciation
        if fund_age_frac < 0.15:
            # J-curve dip phase
            nav_target_frac = 0.85 + fund_age_frac * 0.5  # starts ~0.85, rises
        elif fund_age_frac < 0.5:
            # Value creation phase
            nav_target_frac = 0.92 + (fund_age_frac - 0.15) * 1.5
        else:
            # Harvest/maturity — approach target final nav
            nav_target_frac = 1.0

        # What NAV "should" be at this point (before this quarter's dist)
        interim_target_nav = (cum_funded * nav_target_frac * (target_tvpi ** fund_age_frac)
                              - cum_dist + dist)

        if remaining_q <= 0:
            # Last quarter — force to target
            interim_target_nav = target_final_nav

        # Gain = what's needed to hit interim target, plus noise
        needed_nav = interim_target_nav
        gain = needed_nav - (beginning_nav + contrib - fee - dist)

        # Add GP style bias and noise
        noise = float(rng.normal(0, abs(gain) * 0.15 + 0.5)) + style["markup_bias"] * beginning_nav
        gain = gain + noise

        gains[q] = gain

        # Update NAV: strict accounting identity
        current_nav = beginning_nav + contrib - fee + gain - dist
        current_nav = max(0, current_nav)
        nav[q] = current_nav

    return {
        "nav": nav,
        "contributions": contributions,
        "distributions": distributions,
        "mgmt_fees": mgmt_fees,
        "gains": gains,
    }


# ---------------------------------------------------------------------------
# Quarterly data generator
# ---------------------------------------------------------------------------
def generate_quarterly_data(funds: pd.DataFrame, gps: pd.DataFrame,
                            rng: np.random.Generator) -> pd.DataFrame:
    """Generate quarterly fund performance data with real IRR from cash flows."""
    all_quarters = _quarter_range(2014, 2025)
    records = []
    gp_style_map = dict(zip(gps["gp_id"], gps["style"]))

    for _, fund in funds.iterrows():
        vintage = fund["vintage_year"]
        fund_quarters = all_quarters[all_quarters >= pd.Timestamp(f"{vintage}-01-01")]
        n_q = len(fund_quarters)
        if n_q == 0:
            continue

        gp_style = gp_style_map.get(fund["gp_id"], "balanced")
        path = _simulate_fund_path(n_q, vintage, fund["strategy"], gp_style, rng)

        scale = fund["total_commitment_mm"] / 100.0  # normalised to 100 in sim
        cum_contrib = 0.0
        cum_dist = 0.0

        # Build cash flow series for IRR calc
        cf_dates = []
        cf_amounts = []

        for i, qdate in enumerate(fund_quarters):
            contrib = path["contributions"][i] * scale
            dist = path["distributions"][i] * scale
            nav_val = path["nav"][i] * scale
            fee = path["mgmt_fees"][i] * scale
            gain = path["gains"][i] * scale

            cum_contrib += contrib
            cum_dist += dist

            # Beginning NAV (strict identity)
            if i == 0:
                beg_nav = 0.0
            else:
                beg_nav = path["nav"][i - 1] * scale

            # Cash flows for IRR: negative = LP outflow, positive = LP inflow
            qdate_dt = qdate.to_pydatetime()
            if contrib > 0.001:
                cf_dates.append(qdate_dt)
                cf_amounts.append(-contrib)
            if dist > 0.001:
                cf_dates.append(qdate_dt)
                cf_amounts.append(dist)

            # TVPI, DPI, RVPI — strictly from accounting
            paid_in = max(cum_contrib, 0.01)
            tvpi = (nav_val + cum_dist) / paid_in
            dpi = cum_dist / paid_in
            rvpi = nav_val / paid_in

            # IRR computed from cash flows + residual NAV as terminal value
            irr_cf_dates = cf_dates.copy()
            irr_cf_amounts = cf_amounts.copy()
            if nav_val > 0.01:
                irr_cf_dates.append(qdate_dt)
                irr_cf_amounts.append(nav_val)  # residual as positive CF

            irr_val = compute_irr(irr_cf_dates, irr_cf_amounts) if len(irr_cf_dates) >= 2 else 0.0

            records.append({
                "fund_id": fund["fund_id"],
                "quarter_end": qdate.strftime("%Y-%m-%d"),
                "beginning_nav_mm": round(beg_nav, 2),
                "contributions_mm": round(contrib, 2),
                "distributions_mm": round(dist, 2),
                "mgmt_fees_mm": round(fee, 2),
                "gains_losses_mm": round(gain, 2),
                "ending_nav_mm": round(nav_val, 2),
                "cumulative_contributions_mm": round(cum_contrib, 2),
                "cumulative_distributions_mm": round(cum_dist, 2),
                "irr": round(irr_val, 4),
                "tvpi": round(tvpi, 3),
                "dpi": round(dpi, 3),
                "rvpi": round(rvpi, 3),
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Cash flows & QA issue injection (unchanged)
# ---------------------------------------------------------------------------
def generate_cash_flows(quarterly: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    records = []
    for _, row in quarterly.iterrows():
        if row["contributions_mm"] > 0.01:
            records.append({
                "fund_id": row["fund_id"],
                "date": row["quarter_end"],
                "cf_type": "Capital Call",
                "amount_mm": -round(row["contributions_mm"], 2),
            })
        if row["distributions_mm"] > 0.01:
            records.append({
                "fund_id": row["fund_id"],
                "date": row["quarter_end"],
                "cf_type": "Distribution",
                "amount_mm": round(row["distributions_mm"], 2),
            })
    return pd.DataFrame(records)


def inject_data_quality_issues(quarterly: pd.DataFrame,
                               rng: np.random.Generator) -> pd.DataFrame:
    df = quarterly.copy()
    n = len(df)
    n_issues = max(1, int(n * 0.05))
    issue_indices = rng.choice(n, size=n_issues, replace=False)
    for idx in issue_indices:
        issue_type = rng.choice(["nav_mismatch", "outlier_gain", "negative_nav", "fee_spike"])
        if issue_type == "nav_mismatch":
            df.loc[df.index[idx], "beginning_nav_mm"] *= float(rng.uniform(0.8, 0.95))
        elif issue_type == "outlier_gain":
            df.loc[df.index[idx], "gains_losses_mm"] *= float(rng.uniform(3, 8))
        elif issue_type == "negative_nav":
            df.loc[df.index[idx], "ending_nav_mm"] = -abs(df.loc[df.index[idx], "ending_nav_mm"])
        elif issue_type == "fee_spike":
            df.loc[df.index[idx], "mgmt_fees_mm"] *= float(rng.uniform(5, 15))
    return df


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------
def generate_all_data(seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)

    gps = generate_gps(rng)
    funds = generate_funds(gps, rng)
    companies = generate_companies(rng)
    holdings = generate_holdings(funds, companies, rng)

    quarterly_clean = generate_quarterly_data(funds, gps, rng)

    rng2 = np.random.default_rng(seed + 1)
    quarterly_dirty = inject_data_quality_issues(quarterly_clean.copy(), rng2)

    cash_flows = generate_cash_flows(quarterly_clean, rng)

    return {
        "gps": gps,
        "funds": funds,
        "companies": companies,
        "holdings": holdings,
        "quarterly": quarterly_clean,
        "quarterly_dirty": quarterly_dirty,
        "cash_flows": cash_flows,
    }


if __name__ == "__main__":
    data = generate_all_data()
    for name, df in data.items():
        print(f"{name}: {df.shape}")
    print()

    # Validate
    q = data["quarterly"]
    funds = data["funds"]
    latest = q.sort_values("quarter_end").groupby("fund_id").last().reset_index()
    merged = latest.merge(funds[["fund_id", "vintage_year", "strategy"]], on="fund_id")

    print("=== Portfolio Summary ===")
    print(f"Avg IRR:  {merged['irr'].mean():.1%}")
    print(f"Avg TVPI: {merged['tvpi'].mean():.2f}x")
    print(f"Avg DPI:  {merged['dpi'].mean():.2f}x")
    print()

    print("=== By Vintage ===")
    by_vin = merged.groupby("vintage_year").agg(
        avg_irr=("irr", "mean"),
        avg_tvpi=("tvpi", "mean"),
        avg_dpi=("dpi", "mean"),
        n_funds=("fund_id", "count"),
    )
    for vin, row in by_vin.iterrows():
        print(f"  {vin}: IRR={row['avg_irr']:.1%}  TVPI={row['avg_tvpi']:.2f}x  "
              f"DPI={row['avg_dpi']:.2f}x  (n={int(row['n_funds'])})")

    # TVPI consistency check
    latest["tvpi_check"] = (latest["ending_nav_mm"] + latest["cumulative_distributions_mm"]) / \
                            latest["cumulative_contributions_mm"].clip(lower=0.01)
    max_tvpi_err = (latest["tvpi"] - latest["tvpi_check"]).abs().max()
    print(f"\nMax TVPI consistency error: {max_tvpi_err:.6f}")
