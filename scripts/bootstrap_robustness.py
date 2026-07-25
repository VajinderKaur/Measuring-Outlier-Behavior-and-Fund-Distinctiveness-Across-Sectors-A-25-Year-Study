"""
Bootstrap robustness: resample L2 norms, re-run clustering & Hamming.
Computes confidence intervals and selection frequencies.
"""

import os
import pandas as pd
import numpy as np
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
L2_FILE = os.path.join(BASE_DIR, "l2", "combined_l2.csv")  # We'll create this from create_clusters.py's combined_df
CLUSTER_DIR = os.path.join(BASE_DIR, "clusters")
OUT_DIR = os.path.join(BASE_DIR, "plots", "bootstrap_results")
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Functions (copied from your scripts)
# -----------------------------
def hamming(s1, s2):
    mask = s1.notna() & s2.notna()
    if mask.sum() == 0:
        return np.nan, 0
    mismatch = (s1[mask] != s2[mask]).sum()
    return mismatch, mask.sum()

def portfolio_index_weighted(trajs, weights=None):
    funds = list(trajs.columns)
    m = len(funds)
    if m < 2:
        return 0
    if weights is None:
        weights = [1/m] * m
    total = 0
    for i, f1 in enumerate(funds):
        for j, f2 in enumerate(funds):
            if i < j:
                d, _ = hamming(trajs[f1], trajs[f2])
                if not np.isnan(d):
                    total += weights[i] * weights[j] * d
    return total

def compute_marginal_impacts(traj):
    """Return DataFrame with marginal impacts."""
    funds = list(traj.columns)
    I_full = portfolio_index_weighted(traj)
    impacts = []
    for fund in funds:
        reduced = traj.drop(columns=fund)
        I_red = portfolio_index_weighted(reduced)
        impacts.append({'Fund': fund, 'marginal_impact': I_full - I_red})
    df = pd.DataFrame(impacts).set_index('Fund')
    df['marginal_impact'] = df['marginal_impact'].round(6)
    return df, I_full

# -----------------------------
# Load combined L2 data (we need to create it first)
# -----------------------------
# If not exists, we can load all sector files and combine.
if not os.path.exists(L2_FILE):
    print("Combined L2 file not found. Creating from sector files...")
    import glob
    all_l2 = []
    for f in glob.glob(os.path.join(BASE_DIR, "l2", "*_l2.csv")):
        sector = os.path.basename(f).replace("_l2.csv", "")
        df = pd.read_csv(f).dropna(subset=['L2_Norm'])
        all_l2.append(df)
    combined_df = pd.concat(all_l2, ignore_index=True)
    combined_df.to_csv(L2_FILE, index=False)
else:
    combined_df = pd.read_csv(L2_FILE)

print(f"Total fund-year observations: {len(combined_df)}")
years = sorted(combined_df['Year'].unique())
print(f"Years: {years[0]} - {years[-1]} (n={len(years)})")

# -----------------------------
# Bootstrap parameters
# -----------------------------
B = 2000
BLOCK_LENGTH = 5  # expected block length in years

# Store results
all_marginal_impacts = []  # list of DataFrames, one per bootstrap
all_best_funds = []

print(f"\nRunning {B} bootstrap iterations (block length={BLOCK_LENGTH})...")
for i in range(B):
    if (i+1) % 200 == 0:
        print(f"  Iteration {i+1}/{B}")
    
    # Block bootstrap: sample blocks of years with replacement
    n_years = len(years)
    n_blocks = int(np.ceil(n_years / BLOCK_LENGTH))
    # Randomly select block start indices
    start_indices = np.random.choice(np.arange(0, max(1, n_years - BLOCK_LENGTH + 1)), size=n_blocks, replace=True)
    sampled_years = []
    for start in start_indices:
        sampled_years.extend(years[start:start+BLOCK_LENGTH])
    sampled_years = sampled_years[:n_years]  # trim to original length
    
    # Create bootstrapped dataset
    boot_df = combined_df[combined_df['Year'].isin(sampled_years)].copy()
    # If some years appear multiple times, we need to duplicate observations accordingly.
    # But since we sampled years with replacement, we need to replicate rows for repeated years.
    # Better: create a new DataFrame by concatenating for each sampled year.
    boot_list = []
    for yr in sampled_years:
        boot_list.append(combined_df[combined_df['Year'] == yr])
    boot_df = pd.concat(boot_list, ignore_index=True)
    
    # Now compute deciles on this bootstrapped sample (global across funds and years)
    try:
        boot_df['Cluster'] = pd.qcut(boot_df['L2_Norm'], q=10, labels=range(1, 11), duplicates='drop')
    except:
        # If duplicates cause issues, use rank-based
        boot_df['Cluster'] = pd.qcut(boot_df['L2_Norm'].rank(method='first'), q=10, labels=range(1, 11))
    
    # Convert to trajectory matrix per sector? We need sector-level results.
    # The paper reports sector-level most distinctive fund. We'll compute per sector.
    sector_best = {}
    for sector in boot_df['Sector'].unique():
        sector_df = boot_df[boot_df['Sector'] == sector]
        # Remove SPY if present
        sector_df = sector_df[sector_df['Fund'] != 'SPY']
        if sector_df.empty:
            continue
        # Create trajectory matrix
        yearly = sector_df.groupby(['Year', 'Fund'], as_index=False)['Cluster'].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
        traj = yearly.pivot(index='Year', columns='Fund', values='Cluster').sort_index()
        if traj.shape[1] < 2:
            continue
        impacts, _ = compute_marginal_impacts(traj)
        best_fund = impacts['marginal_impact'].idxmax()
        sector_best[sector] = best_fund
        # Store impacts for this sector
        impacts['bootstrap_iter'] = i
        impacts['sector'] = sector
        all_marginal_impacts.append(impacts.reset_index())
    
    all_best_funds.append(sector_best)

# -----------------------------
# Aggregate results
# -----------------------------
# Marginal impacts: compute percentiles per sector and fund
impact_df = pd.concat(all_marginal_impacts, ignore_index=True)
impact_summary = impact_df.groupby(['sector', 'Fund'])['marginal_impact'].agg([
    ('mean', 'mean'),
    ('std', 'std'),
    ('p2.5', lambda x: x.quantile(0.025)),
    ('p97.5', lambda x: x.quantile(0.975))
]).reset_index()

impact_summary.to_csv(os.path.join(OUT_DIR, "bootstrap_marginal_impacts.csv"), index=False)

# Selection frequencies: count how often each fund is best in each sector
best_freq = pd.DataFrame(all_best_funds)
freq_df = best_freq.apply(pd.Series.value_counts).fillna(0) / B * 100
freq_df.to_csv(os.path.join(OUT_DIR, "selection_frequencies.csv"))
print("\nSelection frequencies (%):")
print(freq_df.round(1))

# -----------------------------
# Summary table for paper
# -----------------------------
summary_table = []
for sector in freq_df.columns:
    top_funds = freq_df[sector].sort_values(ascending=False).head(3)
    for fund, pct in top_funds.items():
        # get confidence interval for that fund in that sector
        ci = impact_summary[(impact_summary['sector']==sector) & (impact_summary['Fund']==fund)]
        if not ci.empty:
            ci_low = ci['p2.5'].values[0]
            ci_high = ci['p97.5'].values[0]
            mean_impact = ci['mean'].values[0]
        else:
            ci_low, ci_high, mean_impact = np.nan, np.nan, np.nan
        summary_table.append({
            'Sector': sector,
            'Fund': fund,
            'Selection Freq (%)': pct,
            'Mean Marginal Impact': mean_impact,
            '95% CI Lower': ci_low,
            '95% CI Upper': ci_high
        })
summary_df = pd.DataFrame(summary_table)
summary_df.to_csv(os.path.join(OUT_DIR, "bootstrap_summary.csv"), index=False)
print("\nBootstrap summary saved.")
print("Done.")