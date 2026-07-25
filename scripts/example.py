import os
import pandas as pd
import numpy as np

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

FUNDS_DIR   = os.path.join(BASE_DIR, "funds")
CLUSTER_DIR = os.path.join(BASE_DIR, "clusters")
SPY_PATH    = os.path.join(BASE_DIR, "benchmarks", "SPY_1999_2025.csv")
OUT_DIR     = os.path.join(BASE_DIR, "portfolio_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# CONFIG
# -----------------------------
RISK_FREE_RATE = 0.04  # 4% annual
BENCHMARK_CLUSTER = 0  # SPY residual against itself is zero

portfolios = {
    "P1": {"SPY": 0.50, "PRNEX": 0.25, "PRHSX": 0.25},
    "P2": {"SPY": 0.50, "VGENX": 0.25, "VGHCX": 0.25},
    "P3": {"SPY": 0.50, "PRNEX": 0.25, "VGHCX": 0.25},
    "P4": {"SPY": 0.50, "VGENX": 0.25, "PRHSX": 0.25},
}

# -----------------------------
# STEP 1 — LOAD ANNUAL RETURNS
# -----------------------------
def load_spy_annual():
    df = pd.read_csv(SPY_PATH, parse_dates=["Date"])
    df = df[["Date", "Daily_Return"]].dropna()
    df["Year"] = df["Date"].dt.year

    annual = (
        df.groupby("Year")["Daily_Return"]
        .apply(lambda x: (1 + x).prod() - 1)
        * 100
    )

    return annual.rename("SPY")


def load_sector_annual(sector):
    path = os.path.join(FUNDS_DIR, sector, "annual", "fund_annual_returns.csv")
    df = pd.read_csv(path, index_col=0)

    df.index = df.index.astype(int)
    df.index.name = "Year"
    df.columns = [c.upper() for c in df.columns]

    # If returns are decimals, convert to percent
    if df.mean().mean() < 2:
        df = df * 100

    return df


print("Loading annual returns...")

spy_annual        = load_spy_annual()
energy_annual     = load_sector_annual("energy")
healthcare_annual = load_sector_annual("healthcare")

annual_parts = {"SPY": spy_annual}

for fund in ["PRNEX", "VGENX"]:
    annual_parts[fund] = energy_annual[fund]

for fund in ["PRHSX", "VGHCX"]:
    annual_parts[fund] = healthcare_annual[fund]

annual_df = pd.concat(annual_parts.values(), axis=1, join="inner")
annual_df.columns = list(annual_parts.keys())
annual_df = annual_df.sort_index()

print(
    f"\nAligned annual returns: {len(annual_df)} years "
    f"({annual_df.index[0]} → {annual_df.index[-1]})"
)
print(annual_df.round(2).to_string())

# -----------------------------
# STEP 2 — LOAD CLUSTER ASSIGNMENTS
# -----------------------------
def load_clusters(sector, allowed_funds=None):
    path = os.path.join(CLUSTER_DIR, f"{sector}_quantiles_global.csv")
    df = pd.read_csv(path)

    df["Year"] = df["Year"].astype(int)
    df["Fund"] = df["Fund"].astype(str).str.upper().str.strip()
    df["Decile_Cluster"] = pd.to_numeric(df["Decile_Cluster"], errors="coerce")

    df = df.dropna(subset=["Year", "Fund", "Decile_Cluster"])

    if allowed_funds is not None:
        allowed_funds = [f.upper() for f in allowed_funds]
        df = df[df["Fund"].isin(allowed_funds)]

    cluster_mode = (
        df.groupby(["Year", "Fund"])["Decile_Cluster"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
        .rename(columns={"Decile_Cluster": "Cluster"})
    )

    cluster_mode["Sector"] = sector

    return cluster_mode


print("\nLoading cluster assignments...")

energy_clusters = load_clusters("energy", allowed_funds=["PRNEX", "VGENX"])
healthcare_clusters = load_clusters("healthcare", allowed_funds=["PRHSX", "VGHCX"])

all_clusters = pd.concat(
    [energy_clusters, healthcare_clusters],
    ignore_index=True
)

duplicate_pairs = (
    all_clusters
    .groupby(["Year", "Fund"])
    .size()
    .reset_index(name="n")
)

duplicate_pairs = duplicate_pairs[duplicate_pairs["n"] > 1]

if not duplicate_pairs.empty:
    print("\nWARNING: Duplicate Year-Fund pairs found after sector filtering:")
    print(duplicate_pairs.head(50).to_string(index=False))

    print("\nFull duplicate rows:")
    print(
        all_clusters[
            all_clusters.duplicated(["Year", "Fund"], keep=False)
        ]
        .sort_values(["Fund", "Year"])
        .to_string(index=False)
    )

    all_clusters = (
        all_clusters
        .groupby(["Year", "Fund"])["Cluster"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
    )
else:
    all_clusters = all_clusters[["Year", "Fund", "Cluster"]]

cluster_pivot = all_clusters.pivot(
    index="Year",
    columns="Fund",
    values="Cluster"
)

# SPY is the benchmark.
# SPY regressed on itself has zero residual, so assign benchmark cluster 0.
cluster_pivot["SPY"] = BENCHMARK_CLUSTER

# Keep only years that are in annual_df
cluster_pivot = cluster_pivot.reindex(annual_df.index)

print("\nCluster assignments:")
print(cluster_pivot.to_string())

# -----------------------------
# STEP 3 — HAMMING DISTANCE
# -----------------------------
def hamming(s1, s2):
    mask = s1.notna() & s2.notna()

    if mask.sum() == 0:
        return np.nan, 0

    mismatch = (s1[mask] != s2[mask]).sum()

    return int(mismatch), int(mask.sum())


# -----------------------------
# STEP 4 — HAMMING INDEX
# -----------------------------
def compute_hamming_index(weights, cluster_pivot):
    fund_list = list(weights.keys())
    w = list(weights.values())

    distances = {}
    index_val = 0.0

    for i in range(len(fund_list)):
        for j in range(i + 1, len(fund_list)):
            f1, f2 = fund_list[i], fund_list[j]

            if f1 not in cluster_pivot.columns or f2 not in cluster_pivot.columns:
                raise KeyError(f"Missing cluster column for {f1} or {f2}")

            d, n = hamming(cluster_pivot[f1], cluster_pivot[f2])
            distances[(f1, f2)] = (d, n)

            if not np.isnan(d):
                index_val += w[i] * w[j] * d

    return index_val, distances


# -----------------------------
# STEP 5 — FUND SUMMARY STATS
# -----------------------------
fund_summary = pd.DataFrame({
    "Mean_Return_%": annual_df.mean(),
    "Std_Return_%":  annual_df.std(),
    "Min_Return_%":  annual_df.min(),
    "Max_Return_%":  annual_df.max(),
    "N_Years":       annual_df.count(),
})

print("\nFund summary statistics:")
print(fund_summary.round(4).to_string())

# -----------------------------
# STEP 6 — CORRELATION MATRIX
# -----------------------------
corr_matrix = annual_df.corr()

print("\nCorrelation matrix annual returns:")
print(corr_matrix.round(4).to_string())

# -----------------------------
# STEP 7 — PORTFOLIO METRICS
# -----------------------------
results = []
hamming_details = {}

for p_name, weights in portfolios.items():

    funds = list(weights.keys())
    w = np.array([weights[f] for f in funds])

    I_p, distances = compute_hamming_index(weights, cluster_pivot)
    hamming_details[p_name] = distances

    port_annual = annual_df[funds]

    mu = port_annual.mean().values
    cov = port_annual.cov().values

    R_p = float(w @ mu)
    var_p = float(w @ cov @ w)
    std_p = float(np.sqrt(var_p))

    port_series = (port_annual * w).sum(axis=1)

    sharpe = (R_p - RISK_FREE_RATE * 100) / std_p

    cumulative = (1 + port_series / 100).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd = float(drawdown.min() * 100)

    cv = std_p / R_p if R_p != 0 else np.nan
    ret_per_div = R_p / I_p if I_p > 0 else np.nan

    results.append({
        "Portfolio":        p_name,
        "Funds":            ", ".join(funds),
        "Return_%":         round(R_p, 4),
        "Std_Dev_%":        round(std_p, 4),
        "Sharpe_Ratio":     round(sharpe, 4),
        "Max_Drawdown_%":   round(max_dd, 4),
        "CV":               round(cv, 4),
        "Hamming_Index":    round(I_p, 4),
        "Return/Diversity": round(ret_per_div, 4),
    })

results_df = pd.DataFrame(results).set_index("Portfolio")

print("\n" + "=" * 70)
print("PORTFOLIO COMPARISON")
print("=" * 70)
print(results_df.to_string())

# -----------------------------
# STEP 8 — SAVE OUTPUTS
# -----------------------------
results_df.to_csv(os.path.join(OUT_DIR, "portfolio_comparison.csv"))
fund_summary.to_csv(os.path.join(OUT_DIR, "fund_summary.csv"))
corr_matrix.to_csv(os.path.join(OUT_DIR, "correlation_matrix.csv"))
annual_df.to_csv(os.path.join(OUT_DIR, "annual_returns_by_fund.csv"))
cluster_pivot.to_csv(os.path.join(OUT_DIR, "cluster_assignments.csv"))

print(f"\nAll outputs saved to: {OUT_DIR}")

# -----------------------------
# STEP 9 — LATEX-READY OUTPUT
# -----------------------------
print("\n" + "=" * 70)
print("LATEX-READY OUTPUT")
print("=" * 70)

print("\n% Fund-level statistics:")
for fund in annual_df.columns:
    row = fund_summary.loc[fund]
    print(
        f"%   {fund}: mean = {row['Mean_Return_%']:.2f}\\%, "
        f"std = {row['Std_Return_%']:.2f}\\%, "
        f"n = {int(row['N_Years'])} years"
    )

print("\n% Hamming distance breakdown per portfolio:")
for p_name, distances in hamming_details.items():
    funds = list(portfolios[p_name].keys())
    w = list(portfolios[p_name].values())
    I_p = results_df.loc[p_name, "Hamming_Index"]

    terms = []
    breakdown = []

    pairs = [
        (funds[i], funds[j])
        for i in range(len(funds))
        for j in range(i + 1, len(funds))
    ]

    for (f1, f2), wi, wj in zip(
        pairs,
        [w[0], w[0], w[1]],
        [w[1], w[2], w[2]]
    ):
        d, n = distances[(f1, f2)]
        coef = wi * wj
        terms.append(f"{coef}({d})")
        breakdown.append(f"{coef * d:.4f}")

    print(
        f"%   I({p_name}) = {' + '.join(terms)} "
        f"= {' + '.join(breakdown)} = {I_p}"
    )

print("\n% Hamming matrices:")
for p_name, weights in portfolios.items():
    funds = list(weights.keys())
    print(f"\n%   M({p_name}): funds = {funds}")

    for i, f1 in enumerate(funds):
        row_vals = []

        for j, f2 in enumerate(funds):
            if i == j:
                row_vals.append("0")
            else:
                pair = (f1, f2) if (f1, f2) in hamming_details[p_name] else (f2, f1)
                d, _ = hamming_details[p_name][pair]
                row_vals.append(str(d))

        print(f"%     {f1}: {' & '.join(row_vals)}")

print("\n% Portfolio comparison table:")
print(
    "% Portfolio & Funds & $R(P)$ & $\\sigma(P)$ & Sharpe "
    "& Max DD & CV & $I(P)$ & $R/I$ \\\\"
)

for p, row in results_df.iterrows():
    print(
        f"${p}$ & {row['Funds']} & "
        f"{row['Return_%']:.2f}\\% & "
        f"{row['Std_Dev_%']:.2f}\\% & "
        f"{row['Sharpe_Ratio']:.4f} & "
        f"{row['Max_Drawdown_%']:.2f}\\% & "
        f"{row['CV']:.4f} & "
        f"{row['Hamming_Index']} & "
        f"{row['Return/Diversity']:.4f} \\\\"
    )