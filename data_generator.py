"""
PE Portfolio Analytics — Synthetic Data Generator
==================================================
Generates realistic private equity fund data starting from CASH FLOWS.

Methodology:
1. Generate capital call schedule (heavy years 1-4, tapering after)
2. Generate distribution schedule (starting year 4-5, accelerating years 6-10)
3. Compute quarterly unrealised gains/losses following J-curve lifecycle
4. Derive NAV each quarter: prev_NAV + calls + gains - distributions
5. Compute IRR from actual cash flow series via XIRR (scipy.optimize.brentq)
6. Compute TVPI = (cumulative_distributions + NAV) / cumulative_calls
7. All metrics are derived, never randomly generated

Strategy allocation targets:
  Buyout 30-35%, Growth Equity 15-20%, VC 10-15%,
  Real Estate 10-15%, Infrastructure 8-12%, Distressed 5-8%

Portfolio company overlap: 10-20% (not 43%)
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

# Strategy pool with explicit probabilities
STRATEGIES = ["Buyout", "Growth Equity", "Venture Capital",
              "Real Estate", "Infrastructure", "Distressed / Special Sits"]
STRATEGY_WEIGHTS = [0.33, 0.18, 0.12, 0.13, 0.10, 0.07]
# Normalise weights (they don't sum to 1.0 because of rounding — intentional buffer)
STRATEGY_WEIGHTS = [w / sum(STRATEGY_WEIGHTS) for w in STRATEGY_WEIGHTS]

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
# Vintage return targets
# ---------------------------------------------------------------------------
VINTAGE_PROFILES = {
    2014: {"tvpi_lo": 1.8, "tvpi_hi": 2.5, "dpi_frac_lo": 0.65, "dpi_frac_hi": 0.85},
    2015: {"tvpi_lo": 1.8, "tvpi_hi": 2.5, "dpi_frac_lo": 0.60, "dpi_frac_hi": 0.80},
    2016: {"tvpi_lo": 1.8, "tvpi_hi": 2.4, "dpi_frac_lo": 0.55, "dpi_frac_hi": 0.75},
    2017: {"tvpi_lo": 1.5, "tvpi_hi": 2.0, "dpi_frac_lo": 0.40, "dpi_frac_hi": 0.60},
    2018: {"tvpi_lo": 1.5, "tvpi_hi": 2.0, "dpi_frac_lo": 0.35, "dpi_frac_hi": 0.55},
    2019: {"tvpi_lo": 1.3, "tvpi_hi": 1.6, "dpi_frac_lo": 0.20, "dpi_frac_hi": 0.40},
    2020: {"tvpi_lo": 1.3, "tvpi_hi": 1.6, "dpi_frac_lo": 0.15, "dpi_frac_hi": 0.35},
    2021: {"tvpi_lo": 1.0, "tvpi_hi": 1.3, "dpi_frac_lo": 0.05, "dpi_frac_hi": 0.15},
    2022: {"tvpi_lo": 1.0, "tvpi_hi": 1.3, "dpi_frac_lo": 0.00, "dpi_frac_hi": 0.08},
    2023: {"tvpi_lo": 0.80, "tvpi_hi": 1.0, "dpi_frac_lo": 0.00, "dpi_frac_hi": 0.03},
}

# Strategy modifiers
STRATEGY_MODS = {
    "Buyout":                    {"tvpi_mult": 1.00},
    "Growth Equity":             {"tvpi_mult": 1.05},
    "Venture Capital":           {"tvpi_mult": 1.10},
    "Distressed / Special Sits": {"tvpi_mult": 0.95},
    "Real Estate":               {"tvpi_mult": 0.95},
    "Infrastructure":            {"tvpi_mult": 0.93},
}

# GP behavioral archetypes
GP_STYLES = {
    "aggressive":    {"markup_bias": 0.008, "dist_delay": -1},
    "conservative":  {"markup_bias": -0.005, "dist_delay": 1},
    "balanced":      {"markup_bias": 0.000, "dist_delay": 0},
}


# ---------------------------------------------------------------------------
# IRR from cash flows (XIRR)
# ---------------------------------------------------------------------------
def compute_irr(dates: list, cashflows: list) -> float:
    """Compute annualised IRR from dated cash flows via XIRR."""
    if not cashflows or len(cashflows) < 2:
        return 0.0
    if all(cf == 0 for cf in cashflows):
        return 0.0

    d0 = dates[0]
    year_fracs = [(d - d0).days / 365.25 for d in dates]

    def npv(rate):
        return sum(cf / (1 + rate) ** t for cf, t in zip(cashflows, year_fracs))

    try:
        return brentq(npv, -0.5, 5.0, maxiter=200)
    except (ValueError, RuntimeError):
        try:
            return brentq(npv, -0.3, 2.0, maxiter=200)
        except (ValueError, RuntimeError):
            return 0.0


# ---------------------------------------------------------------------------
# Quarter range
# ---------------------------------------------------------------------------
def _quarter_range(start_year: int, end_year: int) -> pd.DatetimeIndex:
    return pd.date_range(start=f"{start_year}-03-31", end=f"{end_year}-12-31", freq="QE")


# ---------------------------------------------------------------------------
# Entity generators
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
    # Pre-build strategy sequence for ~30 funds to hit allocation targets exactly
    # Buyout 32%, Growth 18%, VC 12%, RE 12%, Infra 10%, Distressed 7%
    _strat_sequence = (
        ["Buyout"] * 10 +
        ["Growth Equity"] * 6 +
        ["Venture Capital"] * 4 +
        ["Real Estate"] * 4 +
        ["Infrastructure"] * 3 +
        ["Distressed / Special Sits"] * 2
    )
    # Pad with non-buyout to avoid buyout overweight
    _strat_sequence += ["Growth Equity", "Infrastructure", "Real Estate",
                         "Venture Capital", "Distressed / Special Sits"]
    # Use a separate rng to shuffle so it doesn't affect other generators
    strat_rng = np.random.default_rng(123)
    strat_rng.shuffle(_strat_sequence)

    records = []
    fund_idx = 0
    for vintage in range(2014, 2024):
        n_funds = int(rng.integers(2, 5))
        for _ in range(n_funds):
            gp = gps.iloc[fund_idx % len(gps)]
            strategy = _strat_sequence[fund_idx % len(_strat_sequence)]
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
    """
    Generate fund-company holdings with 10-20% overlap rate.
    """
    records = []
    company_ids = companies["company_id"].tolist()
    n_companies = len(company_ids)

    # Assign companies per fund from non-overlapping pools to control overlap
    # First, partition 170 companies across funds (no overlap)
    # Then designate 30 companies (~15%) as shared across 2 funds
    rng.shuffle(company_ids)
    shared_cos = set(company_ids[:30])   # 15% overlap pool
    unique_cos = company_ids[30:]        # 85% unique pool

    unique_idx = 0
    for _, fund in funds.iterrows():
        n_holdings = int(rng.integers(5, 9))

        # Pick mostly from unique pool
        n_unique = min(n_holdings, len(unique_cos) - unique_idx)
        chosen = list(unique_cos[unique_idx:unique_idx + n_unique])
        unique_idx += n_unique

        for cid in chosen:
            cost = round(float(rng.uniform(10, 150)), 1)
            records.append({
                "fund_id": fund["fund_id"],
                "company_id": cid,
                "initial_cost_mm": cost,
                "ownership_pct": round(float(rng.uniform(0.02, 0.25)), 3),
            })

    # Now assign shared companies to exactly 2 funds each
    fund_ids = funds["fund_id"].tolist()
    existing_pairs = set((r["fund_id"], r["company_id"]) for r in records)

    for cid in shared_cos:
        assigned_funds = rng.choice(fund_ids, size=2, replace=False)
        for fid in assigned_funds:
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
# Core: cash-flow-first fund simulation
# ---------------------------------------------------------------------------
def _simulate_fund(n_quarters: int, vintage: int, strategy: str,
                   gp_style: str, rng: np.random.Generator) -> dict:
    """
    Simulate a PE fund lifecycle starting from cash flows.

    Phase 1 (Q1-Q16, years 1-4): Investment period
      - Heavy capital calls, front-loaded
      - Small negative/flat unrealised returns (J-curve)

    Phase 2 (Q12-Q28, years 3-7): Value creation
      - Capital calls taper off
      - Increasing positive markups as companies grow
      - Distributions begin (year 4-5)

    Phase 3 (Q20+, years 5+): Harvesting
      - No more capital calls
      - Large distributions as exits happen
      - NAV declining as portfolio is sold down

    All metrics derived from these cash flows.
    """
    vp = VINTAGE_PROFILES[vintage]
    sm = STRATEGY_MODS[strategy]
    style = GP_STYLES[gp_style]

    commitment = 100.0  # normalised

    # ── Target outcomes ────────────────────────────────────────────
    target_tvpi = float(rng.uniform(vp["tvpi_lo"], vp["tvpi_hi"])) * sm["tvpi_mult"]
    target_tvpi = max(0.7, target_tvpi)
    dpi_frac = float(rng.uniform(vp["dpi_frac_lo"], vp["dpi_frac_hi"]))

    # ── STEP 1: Capital call schedule ──────────────────────────────
    # Heavy years 1-4 (Q1-Q16), tapering after
    drawdown_pct = float(rng.uniform(0.88, 0.98))  # draw 88-98% of commitment
    total_to_call = commitment * drawdown_pct

    calls = np.zeros(n_quarters)
    # Investment period: first 14-20 quarters
    invest_end = min(int(rng.integers(14, 20)), n_quarters)

    for q in range(invest_end):
        if q < 4:
            # Year 1: heaviest calls (~30% of total)
            weight = float(rng.uniform(0.06, 0.10))
        elif q < 8:
            # Year 2: still heavy (~25%)
            weight = float(rng.uniform(0.05, 0.08))
        elif q < 12:
            # Year 3: moderate (~20%)
            weight = float(rng.uniform(0.03, 0.06))
        elif q < 16:
            # Year 4: tapering (~15%)
            weight = float(rng.uniform(0.02, 0.04))
        else:
            # Year 5: minimal follow-ons (~10%)
            weight = float(rng.uniform(0.01, 0.03))
        calls[q] = weight

    # Normalise to total drawdown
    if calls.sum() > 0:
        calls = calls / calls.sum() * total_to_call

    # ── STEP 2: Distribution schedule ──────────────────────────────
    # Starts year 4-5, accelerates years 6-10
    total_value = target_tvpi * total_to_call
    total_distributions = dpi_frac * total_value
    target_final_nav = max(0, total_value - total_distributions)

    dist_start_q = int(rng.integers(14, 20)) + style["dist_delay"]
    dist_start_q = max(12, min(dist_start_q, n_quarters - 2))

    dists = np.zeros(n_quarters)
    for q in range(dist_start_q, n_quarters):
        quarters_into_harvest = q - dist_start_q
        harvest_length = max(n_quarters - dist_start_q, 1)
        progress = quarters_into_harvest / harvest_length

        # Bell-shaped: ramp up, peak around 40-60% through harvest, then taper
        weight = np.exp(-((progress - 0.45) ** 2) / 0.12)
        weight += float(rng.uniform(0, 0.15))
        dists[q] = max(0, weight)

    if dists.sum() > 0:
        dists = dists / dists.sum() * total_distributions

    # ── STEP 3: Quarterly unrealised gains/losses ──────────────────
    # Derive gains so that NAV path is internally consistent
    # NAV_q = NAV_{q-1} + calls_q + gains_q - dists_q
    # Final NAV should hit target_final_nav

    nav = np.zeros(n_quarters)
    gains = np.zeros(n_quarters)
    mgmt_fees = np.zeros(n_quarters)

    # Fee schedule: ~0.4-0.5% quarterly on commitment during investment,
    # ~0.3-0.4% on NAV during harvest
    fee_rate_invest = float(rng.uniform(0.004, 0.005))
    fee_rate_harvest = float(rng.uniform(0.003, 0.004))

    # Pre-compute a smooth NAV target curve
    # J-curve: dip years 1-2, value creation years 3-7, harvest years 8+
    cum_calls = np.cumsum(calls)
    cum_dists = np.cumsum(dists)

    current_nav = 0.0
    for q in range(n_quarters):
        age_frac = q / max(n_quarters - 1, 1)

        # Management fee
        if q < invest_end:
            fee = commitment * fee_rate_invest
        else:
            fee = max(current_nav, 0) * fee_rate_harvest
        mgmt_fees[q] = fee

        # What should NAV be at this point?
        if q == n_quarters - 1:
            # Final quarter: hit target
            nav_target = target_final_nav
        else:
            # Smooth interpolation with J-curve shape
            if age_frac < 0.15:
                # J-curve dip: NAV < called capital (fees + early losses)
                value_mult = 0.85 + age_frac * 0.5 + style["markup_bias"] * 2
            elif age_frac < 0.45:
                # Value creation: NAV grows above cost
                creation_progress = (age_frac - 0.15) / 0.30
                value_mult = 0.92 + creation_progress * (target_tvpi - 1.0) * 0.6
            else:
                # Harvest: NAV = remaining value after distributions
                harvest_progress = (age_frac - 0.45) / 0.55
                # Value continues to grow but distributions reduce NAV
                peak_mult = 0.92 + (target_tvpi - 1.0) * 0.6
                value_mult = peak_mult * (1 + harvest_progress * 0.3)

            nav_target = cum_calls[q] * value_mult - cum_dists[q]
            nav_target = max(0, nav_target)

        # Gain = what's needed to reach target, plus small noise
        needed_gain = nav_target - (current_nav + calls[q] - fee - dists[q])
        noise = float(rng.normal(0, abs(needed_gain) * 0.08 + 0.2))
        gain = needed_gain + noise + style["markup_bias"] * current_nav

        gains[q] = gain

        # NAV identity: strictly enforced
        current_nav = current_nav + calls[q] - fee + gain - dists[q]
        current_nav = max(0, current_nav)
        nav[q] = current_nav

    return {
        "nav": nav,
        "contributions": calls,
        "distributions": dists,
        "mgmt_fees": mgmt_fees,
        "gains": gains,
    }


# ---------------------------------------------------------------------------
# Quarterly data with real IRR
# ---------------------------------------------------------------------------
def generate_quarterly_data(funds: pd.DataFrame, gps: pd.DataFrame,
                            rng: np.random.Generator) -> pd.DataFrame:
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
        path = _simulate_fund(n_q, vintage, fund["strategy"], gp_style, rng)

        scale = fund["total_commitment_mm"] / 100.0
        cum_contrib = 0.0
        cum_dist = 0.0

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

            beg_nav = 0.0 if i == 0 else path["nav"][i - 1] * scale

            # Cash flows for IRR
            qdate_dt = qdate.to_pydatetime()
            if contrib > 0.001:
                cf_dates.append(qdate_dt)
                cf_amounts.append(-contrib)
            if dist > 0.001:
                cf_dates.append(qdate_dt)
                cf_amounts.append(dist)

            # Multiples — strictly derived
            paid_in = max(cum_contrib, 0.01)
            tvpi = (nav_val + cum_dist) / paid_in
            dpi = cum_dist / paid_in
            rvpi = nav_val / paid_in

            # IRR from actual cash flows + residual NAV
            irr_dates = cf_dates.copy()
            irr_amounts = cf_amounts.copy()
            if nav_val > 0.01:
                irr_dates.append(qdate_dt)
                irr_amounts.append(nav_val)

            irr_val = compute_irr(irr_dates, irr_amounts) if len(irr_dates) >= 2 else 0.0

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
# Cash flows & QA issue injection
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

    q = data["quarterly"]
    funds = data["funds"]
    latest = q.sort_values("quarter_end").groupby("fund_id").last().reset_index()
    merged = latest.merge(funds[["fund_id", "vintage_year", "strategy"]], on="fund_id")

    print("=== Portfolio Summary ===")
    print(f"  Avg IRR:  {merged['irr'].mean():.1%}")
    print(f"  Avg TVPI: {merged['tvpi'].mean():.2f}x")
    print(f"  Avg DPI:  {merged['dpi'].mean():.2f}x")
    print()

    print("=== By Vintage ===")
    by_vin = merged.groupby("vintage_year").agg(
        avg_irr=("irr", "mean"), avg_tvpi=("tvpi", "mean"),
        avg_dpi=("dpi", "mean"), n=("fund_id", "count"),
    )
    for vin, row in by_vin.iterrows():
        print(f"  {vin}: IRR={row['avg_irr']:6.1%}  TVPI={row['avg_tvpi']:.2f}x  "
              f"DPI={row['avg_dpi']:.2f}x  (n={int(row['n'])})")

    print("\n=== Strategy Allocation ===")
    strat = funds["strategy"].value_counts(normalize=True).sort_index()
    for s, pct in strat.items():
        print(f"  {s}: {pct:.0%}")

    print("\n=== Overlap Rate ===")
    h = data["holdings"]
    fund_counts = h.groupby("company_id")["fund_id"].nunique()
    overlap = (fund_counts >= 2).sum()
    print(f"  Companies in 2+ funds: {overlap}/{len(data['companies'])} = {overlap/len(data['companies']):.1%}")

    # TVPI consistency
    latest["tvpi_check"] = (latest["ending_nav_mm"] + latest["cumulative_distributions_mm"]) / \
                            latest["cumulative_contributions_mm"].clip(lower=0.01)
    max_err = (latest["tvpi"] - latest["tvpi_check"]).abs().max()
    print(f"\n=== Consistency ===")
    print(f"  Max TVPI error: {max_err:.6f}")
