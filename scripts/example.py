import os
import glob
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

portfolios = {
    "P1": {"VFIAX": 0.50, "PRNEX": 0.25, "PRHSX": 0.25},
    "P2": {"VFIAX": 0.50, "VGENX": 0.25, "VGHCX": 0.25},
    "P3": {"VFIAX": 0.50, "PRNEX": 0.25, "VGHCX": 0.25},
    "P4": {"VFIAX": 0.50, "VGENX": 0.25, "PRHSX": 0.25},
}

fund_sector_map = {
    "VFIAX": None,          # SPY proxy
    "PRNEX": "energy",
    "VGENX": "energy",
    "PRHSX": "healthcare",
    "VGHCX": "healthcare",
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
    return annual.rename("VFIAX")


def load_sector_annual(sector):
    path = os.path.join(FUNDS_DIR, sector, "annual", "fund_annual_returns.csv")
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(int)
    df.index.name = "Year"
    df.columns = [c.upper() for c in df.columns]
    if df.mean().mean() < 2:
        df = df * 100
    return df


print("Loading annual returns...")
spy_annual        = load_spy_annual()
energy_annual     = load_sector_annual("energy")
healthcare_annual = load_sector_annual("healthcare")

annual_parts = {"VFIAX": spy_annual}
for fund in ["PRNEX", "VGENX"]:
    annual_parts[fund] = energy_annual[fund]
for fund in ["PRHSX", "VGHCX"]:
    annual_parts[fund] = healthcare_annual[fund]

annual_df = pd.concat(annual_parts.values(), axis=1, join="inner")
annual_df.columns = list(annual_parts.keys())
annual_df = annual_df.sort_index()

print(f"\nAligned annual returns: {len(annual_df)} years "
      f"({annual_df.index[0]} → {annual_df.index[-1]})")
print(annual_df.round(2).to_string())

# -----------------------------
# STEP 2 — LOAD CLUSTER ASSIGNMENTS
# -----------------------------
def load_clusters(sector):
    path = os.path.join(CLUSTER_DIR, f"{sector}_quantiles_byyear.csv")
    df = pd.read_csv(path)
    df["Year"]     = df["Year"].astype(int)
    df["Quantile"] = pd.to_numeric(df["Quantile"], errors="coerce")
    df["Fund"]     = df["Fund"].str.upper()
    cluster_mode = (
        df.groupby(["Year", "Fund"])["Quantile"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
        .rename(columns={"Quantile": "Cluster"})
    )
    return cluster_mode

print("\nLoading cluster assignments...")
all_clusters  = pd.concat(
    [load_clusters("energy"), load_clusters("healthcare")],
    ignore_index=True
)
cluster_pivot = all_clusters.pivot(index="Year", columns="Fund", values="Cluster")

# VFIAX = SPY = always cluster 1
cluster_pivot["VFIAX"] = 1

print("\nCluster assignments (mode per year):")
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
# formula: I(P) = w1*w2*d12 + w1*w3*d13 + w2*w3*d23
# with w=(0.5, 0.25, 0.25):
#   w1*w2 = 0.5*0.25 = 0.125
#   w1*w3 = 0.5*0.25 = 0.125
#   w2*w3 = 0.25*0.25 = 0.0625
# -----------------------------
def compute_hamming_index(funds, weights, cluster_pivot):
    fund_list = list(funds.keys())
    w         = list(weights.values())

    distances = {}
    index_val = 0.0

    for i in range(len(fund_list)):
        for j in range(i + 1, len(fund_list)):
            f1, f2 = fund_list[i], fund_list[j]
            d, n   = hamming(cluster_pivot[f1], cluster_pivot[f2])
            distances[(f1, f2)] = (d, n)
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
print("\nCorrelation matrix (annual returns):")
print(corr_matrix.round(4).to_string())

# -----------------------------
# STEP 7 — PORTFOLIO METRICS
# -----------------------------
results      = []
hamming_details = {}   # store distances for LaTeX

for p_name, weights in portfolios.items():

    funds = list(weights.keys())
    w     = np.array([weights[f] for f in funds])

    # --- Hamming index (computed from data) ---
    I_p, distances = compute_hamming_index(weights, weights, cluster_pivot)
    hamming_details[p_name] = distances

    port_annual = annual_df[funds]

    # mean return vector
    mu  = port_annual.mean().values
    cov = port_annual.cov().values

    # portfolio return & risk
    R_p   = float(w @ mu)
    var_p = float(w @ cov @ w)
    std_p = float(np.sqrt(var_p))

    # weighted annual series
    port_series = (port_annual * w).sum(axis=1)

    # Sharpe ratio
    sharpe = (R_p - RISK_FREE_RATE * 100) / std_p

    # max drawdown
    cumulative  = (1 + port_series / 100).cumprod()
    rolling_max = cumulative.cummax()
    drawdown    = (cumulative - rolling_max) / rolling_max
    max_dd      = float(drawdown.min() * 100)

    # coefficient of variation
    cv = std_p / R_p if R_p != 0 else np.nan

    # return per unit of diversification
    ret_per_div = R_p / I_p if I_p > 0 else np.nan

    results.append({
        "Portfolio":        p_name,
        "Funds":            ", ".join(funds),
        "Return_%":         round(R_p,        4),
        "Std_Dev_%":        round(std_p,       4),
        "Sharpe_Ratio":     round(sharpe,      4),
        "Max_Drawdown_%":   round(max_dd,      4),
        "CV":               round(cv,          4),
        "Hamming_Index":    round(I_p,         4),
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
    print(f"%   {fund}: mean = {row['Mean_Return_%']:.2f}\\%,  "
          f"std = {row['Std_Return_%']:.2f}\\%,  "
          f"n = {int(row['N_Years'])} years")

print("\n% Hamming distance breakdown per portfolio:")
for p_name, distances in hamming_details.items():
    funds     = list(portfolios[p_name].keys())
    w         = list(portfolios[p_name].values())
    I_p       = results_df.loc[p_name, "Hamming_Index"]
    terms     = []
    breakdown = []
    pairs     = [(funds[i], funds[j])
                 for i in range(len(funds))
                 for j in range(i+1, len(funds))]
    for (f1, f2), wi, wj in zip(
        pairs,
        [w[0], w[0], w[1]],
        [w[1], w[2], w[2]]
    ):
        d, n = distances[(f1, f2)]
        coef = wi * wj
        terms.append(f"{coef}({d})")
        breakdown.append(f"{coef * d:.4f}")
    print(f"%   I({p_name}) = {' + '.join(terms)} "
          f"= {' + '.join(breakdown)} = {I_p}")

print("\n% Hamming matrices (fill into LaTeX bmatrix):")
for p_name, weights in portfolios.items():
    funds = list(weights.keys())
    print(f"\n%   M({p_name}): funds = {funds}")
    for i, f1 in enumerate(funds):
        row_vals = []
        for j, f2 in enumerate(funds):
            if i == j:
                row_vals.append("0")
            else:
                pair = (f1, f2) if (f1, f2) in hamming_details[p_name] \
                       else (f2, f1)
                d, _ = hamming_details[p_name][pair]
                row_vals.append(str(d))
        print(f"%     {f1}: {' & '.join(row_vals)}")

print("\n% Portfolio comparison table:")
print("% Portfolio & Funds & $R(P)$ & $\\sigma(P)$ & Sharpe "
      "& Max DD & CV & $I(P)$ & $R/I$ \\\\")
for p, row in results_df.iterrows():
    print(f"${p}$ & {row['Funds']} & "
          f"{row['Return_%']:.2f}\\% & "
          f"{row['Std_Dev_%']:.2f}\\% & "
          f"{row['Sharpe_Ratio']:.4f} & "
          f"{row['Max_Drawdown_%']:.2f}\\% & "
          f"{row['CV']:.4f} & "
          f"{row['Hamming_Index']} & "
          f"{row['Return/Diversity']:.4f} \\\\")