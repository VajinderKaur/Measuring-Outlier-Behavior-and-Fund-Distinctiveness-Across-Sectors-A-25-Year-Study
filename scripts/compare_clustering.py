"""
compare_clustering.py
Run k-means and Ward clustering on L2 norms (k=10).
SPY and the "combined" file are excluded.
Decile baseline is computed dynamically from decile files.
"""

import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
L2_DIR = os.path.join(BASE_DIR, "l2")
CLUSTER_DIR = os.path.join(BASE_DIR, "clusters")
OUT_DIR = os.path.join(BASE_DIR, "plots", "clustering_comparison")
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
    """Return the fund with the highest marginal impact."""
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
    return best_fund

# -----------------------------
# Function to get decile baseline from decile files
# -----------------------------
def get_decile_baseline(sector):
    """Read the decile file and compute the most distinctive fund."""
    sector_key = sector.lower().replace(' ', '_')  # handle "real estate"
    fpath = os.path.join(CLUSTER_DIR, f"{sector_key}_quantiles_global.csv")
    if not os.path.exists(fpath):
        print(f"  Warning: Decile file not found: {fpath}")
        return None
    df = pd.read_csv(fpath)
    df = df.rename(columns={'Decile_Cluster': 'Cluster'})
    df = df[df['Fund'] != 'SPY']
    yearly = df.groupby(['Year', 'Fund'], as_index=False)['Cluster'].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
    traj = yearly.pivot(index='Year', columns='Fund', values='Cluster').sort_index()
    return compute_most_distinctive(traj)

# -----------------------------
# Load L2 data, EXCLUDING SPY and combined
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
print(f"Loaded {len(combined_df)} fund-year observations (SPY & combined excluded).")

# Get sectors with at least 3 funds
sectors = []
for sector in combined_df['Sector'].unique():
    n_funds = combined_df[combined_df['Sector'] == sector]['Fund'].nunique()
    if n_funds >= 3:
        sectors.append(sector)
    else:
        print(f"Skipping {sector}: only {n_funds} funds")

print(f"Processing sectors: {sectors}")

# -----------------------------
# Run KMeans and Ward on each sector
# -----------------------------
results = {'Sector': [], 'Decile (Baseline)': [], 'KMeans': [], 'Ward': []}

for sector in sectors:
    print(f"\nProcessing {sector}...")
    
    # Get decile baseline from the decile files (matches your Hamming script)
    baseline_fund = get_decile_baseline(sector)
    
    sector_df = combined_df[combined_df['Sector'] == sector].copy()
    X = sector_df[['L2_Norm']].values
    
    # --- K-Means (k=10) ---
    try:
        kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(X)
        sector_df_kmeans = sector_df.copy()
        sector_df_kmeans['Cluster'] = kmeans_labels + 1
        
        yearly_kmeans = sector_df_kmeans.groupby(['Year', 'Fund'], as_index=False)['Cluster'].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
        traj_kmeans = yearly_kmeans.pivot(index='Year', columns='Fund', values='Cluster').sort_index()
        best_kmeans = compute_most_distinctive(traj_kmeans)
    except Exception as e:
        print(f"  KMeans failed: {e}")
        best_kmeans = 'ERROR'
    
    # --- Hierarchical (Ward, k=10) ---
    try:
        ward = AgglomerativeClustering(n_clusters=10, linkage='ward')
        ward_labels = ward.fit_predict(X)
        sector_df_ward = sector_df.copy()
        sector_df_ward['Cluster'] = ward_labels + 1
        
        yearly_ward = sector_df_ward.groupby(['Year', 'Fund'], as_index=False)['Cluster'].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
        traj_ward = yearly_ward.pivot(index='Year', columns='Fund', values='Cluster').sort_index()
        best_ward = compute_most_distinctive(traj_ward)
    except Exception as e:
        print(f"  Ward failed: {e}")
        best_ward = 'ERROR'
    
    # Store sector name in title case
    sector_display = sector.title()
    if sector_display.lower() == 'real estate':
        sector_display = 'Real Estate'
    
    results['Sector'].append(sector_display)
    results['Decile (Baseline)'].append(baseline_fund)
    results['KMeans'].append(best_kmeans)
    results['Ward'].append(best_ward)

# -----------------------------
# Create and display DataFrame
# -----------------------------
df_results = pd.DataFrame(results)

print("\n" + "="*60)
print("CLUSTERING METHOD COMPARISON (Most Distinctive Fund)")
print("SPY and combined file excluded")
print("Decile baseline computed from decile files")
print("="*60)
print(df_results.to_string(index=False))

out_path = os.path.join(OUT_DIR, "clustering_method_comparison.csv")
df_results.to_csv(out_path, index=False)
print(f"\n✓ Saved to: {out_path}")