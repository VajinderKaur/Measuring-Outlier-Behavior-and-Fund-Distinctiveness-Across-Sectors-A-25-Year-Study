"""
rolling_quantiles_robustness.py
Re-run the "most distinctive fund" analysis using expanding-window quantiles.
For each year from 2000 to 2025, compute deciles using only data up to that year.
This tests whether the findings hold when using only historical information.
"""

import os
import pandas as pd
import numpy as np
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
L2_DIR = os.path.join(BASE_DIR, "l2")
CLUSTER_DIR = os.path.join(BASE_DIR, "clusters")
OUT_DIR = os.path.join(BASE_DIR, "plots", "window_quantiles")
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Hamming functions (copied from your script)
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

def compute_most_distinctive(traj):
    funds = list(traj.columns)
    m = len(funds)
    if m < 2:
        return None
    I_full = portfolio_index_weighted(traj)
    impacts = {}
    for fund in funds:
        reduced = traj.drop(columns=fund)
        I_red = portfolio_index_weighted(reduced)
        impacts[fund] = I_full - I_red
    best_fund = max(impacts, key=impacts.get)
    return best_fund, impacts

# -----------------------------
# Load combined L2 data
# -----------------------------
all_l2 = []
for f in os.listdir(L2_DIR):
    if f == 'combined_l2.csv':
        continue
    if f.endswith('_l2.csv'):
        sector = f.replace('_l2.csv', '')
        df = pd.read_csv(os.path.join(L2_DIR, f)).dropna(subset=['L2_Norm'])
        df = df[df['Fund'] != 'SPY']
        df['Sector'] = sector
        all_l2.append(df)
combined_df = pd.concat(all_l2, ignore_index=True)
print(f"Loaded {len(combined_df)} fund-year observations (SPY excluded).")

# Get unique sectors
sectors = []
for sector in combined_df['Sector'].unique():
    n_funds = combined_df[combined_df['Sector'] == sector]['Fund'].nunique()
    if n_funds >= 3:
        sectors.append(sector)
print(f"Processing sectors: {sectors}")

# Years for expansion (start from 2000 to have enough data)
years = sorted(combined_df['Year'].unique())
print(f"Years: {years[0]} - {years[-1]}")

# -----------------------------
# Baseline results for comparison (from pooled quantiles)
# -----------------------------
baseline_results = {}
for sector in sectors:
    fpath = os.path.join(CLUSTER_DIR, f"{sector.lower().replace(' ', '_')}_quantiles_global.csv")
    if os.path.exists(fpath):
        df = pd.read_csv(fpath)
        df = df.rename(columns={'Decile_Cluster': 'Cluster'})
        df = df[df['Fund'] != 'SPY']
        yearly = df.groupby(['Year', 'Fund'], as_index=False)['Cluster'].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
        traj = yearly.pivot(index='Year', columns='Fund', values='Cluster').sort_index()
        best, _ = compute_most_distinctive(traj)
        baseline_results[sector] = best

# -----------------------------
# Expanding-window quantiles
# -----------------------------
print("\n" + "="*60)
print("EXPANDING-WINDOW QUANTILES ANALYSIS")
print("="*60)

window_results = {}  # sector -> {year: best_fund}
window_best_freq = {}  # sector -> fund -> count

# Determine minimum year to start (need at least 3 years of data)
min_start_year = max(years[0] + 5, 2000)  # start from 2000 or later

for sector in sectors:
    print(f"\nProcessing {sector}...")
    sector_df = combined_df[combined_df['Sector'] == sector].copy()
    funds_in_sector = sector_df['Fund'].unique()
    
    sector_window_results = {}
    fund_counts = {fund: 0 for fund in funds_in_sector}
    
    for t in range(min_start_year, years[-1] + 1):
        # Use data up to year t (expanding window)
        historical_df = sector_df[sector_df['Year'] <= t].copy()
        
        # Need at least 3 years of data and 3 funds
        if len(historical_df) < 30 or historical_df['Fund'].nunique() < 3:
            continue
        
        # Compute deciles using only historical data
        try:
            historical_df['Cluster'] = pd.qcut(
                historical_df['L2_Norm'], 
                q=10, 
                labels=range(1, 11),
                duplicates='drop'
            )
        except:
            # Fallback: rank-based if duplicates cause issues
            historical_df['Cluster'] = pd.qcut(
                historical_df['L2_Norm'].rank(method='first'),
                q=10,
                labels=range(1, 11)
            )
        
        # Get modal cluster per year per fund
        yearly = historical_df.groupby(['Year', 'Fund'], as_index=False)['Cluster'].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
        
        # Create trajectory matrix
        traj = yearly.pivot(index='Year', columns='Fund', values='Cluster').sort_index()
        
        if len(traj.columns) < 3:
            continue
        
        # Compute most distinctive fund for this window
        best_fund, impacts = compute_most_distinctive(traj)
        if best_fund is not None:
            sector_window_results[t] = best_fund
            fund_counts[best_fund] = fund_counts.get(best_fund, 0) + 1
    
    window_results[sector] = sector_window_results
    window_best_freq[sector] = fund_counts

# -----------------------------
# Summary: most frequently selected fund across windows
# -----------------------------
print("\n" + "="*60)
print("SUMMARY: MOST FREQUENTLY SELECTED FUND (Expanding Windows)")
print("="*60)

summary_rows = []
for sector in sectors:
    if sector not in window_best_freq:
        continue
    counts = window_best_freq[sector]
    total_windows = len(window_results.get(sector, {}))
    
    # Sort by frequency
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    top_fund = sorted_counts[0][0] if sorted_counts else None
    top_freq = sorted_counts[0][1] if sorted_counts else 0
    top_pct = (top_freq / total_windows * 100) if total_windows > 0 else 0
    
    summary_rows.append({
        'Sector': sector,
        'Most Frequent Fund': top_fund,
        'Frequency (Years)': top_freq,
        'Total Windows': total_windows,
        'Selection Rate (%)': round(top_pct, 1),
        'Baseline (Pooled)': baseline_results.get(sector, 'N/A')
    })
    
    print(f"\n{sector.upper()}:")
    print(f"  Most frequent: {top_fund} ({top_freq}/{total_windows} windows, {top_pct:.1f}%)")
    print(f"  Baseline (pooled): {baseline_results.get(sector, 'N/A')}")
    print(f"  Top 3 funds:")
    for i, (fund, count) in enumerate(sorted_counts[:3]):
        pct = (count / total_windows * 100) if total_windows > 0 else 0
        print(f"    {i+1}. {fund}: {count} ({pct:.1f}%)")

# -----------------------------
# Create comparison table
# -----------------------------
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUT_DIR, "window_quantiles_summary.csv"), index=False)
print(f"\n✓ Summary saved to: {os.path.join(OUT_DIR, 'window_quantiles_summary.csv')}")

# -----------------------------
# Create detailed year-by-year results
# -----------------------------
for sector in sectors:
    if sector not in window_results:
        continue
    results = window_results[sector]
    df_year = pd.DataFrame([
        {'Year': year, 'Best_Fund': fund} 
        for year, fund in results.items()
    ])
    out_path = os.path.join(OUT_DIR, f"{sector.replace(' ', '_')}_window_results.csv")
    df_year.to_csv(out_path, index=False)
    print(f"✓ Detailed results saved to: {out_path}")

print("\n" + "="*60)
print("Done. Window-based quantiles robustness check complete.")
print("="*60)