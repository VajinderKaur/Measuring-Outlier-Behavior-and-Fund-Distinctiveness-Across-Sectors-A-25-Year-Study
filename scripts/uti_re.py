"""
compare_competing_clusters.py
Extract and compare the raw cluster assignments (Decile, KMeans, Ward)
for the competing funds in Utilities and Real Estate.
"""

import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLUSTER_DIR = os.path.join(BASE_DIR, "clusters")
L2_DIR = os.path.join(BASE_DIR, "l2")
OUT_DIR = os.path.join(BASE_DIR, "plots", "competing_clusters")
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Hamming helpers (needed for baseline)
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
    return max(impacts, key=impacts.get)

# -----------------------------
# Decile cluster labels (DataFrame)
# -----------------------------
def get_decile_clusters(sector):
    sector_key = sector.lower()
    for sep in [' ', '_']:
        fname = f"{sector_key.replace(' ', sep)}_quantiles_global.csv"
        fpath = os.path.join(CLUSTER_DIR, fname)
        if os.path.exists(fpath):
            break
    else:
        return None
    df = pd.read_csv(fpath)
    df = df.rename(columns={'Decile_Cluster': 'Decile'})
    df = df[['Fund', 'Year', 'Decile']]
    return df

# -----------------------------
# Baseline (most distinctive fund) – optional
# -----------------------------
def get_decile_baseline(sector):
    sector_key = sector.lower()
    for sep in [' ', '_']:
        fname = f"{sector_key.replace(' ', sep)}_quantiles_global.csv"
        fpath = os.path.join(CLUSTER_DIR, fname)
        if os.path.exists(fpath):
            break
    else:
        return None
    df = pd.read_csv(fpath)
    df = df.rename(columns={'Decile_Cluster': 'Cluster'})
    df = df[df['Fund'] != 'SPY']
    yearly = df.groupby(['Year', 'Fund'], as_index=False)['Cluster'].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
    traj = yearly.pivot(index='Year', columns='Fund', values='Cluster').sort_index()
    return compute_most_distinctive(traj)

# -----------------------------
# KMeans and Ward labels (DataFrame)
# -----------------------------
def run_kmeans_and_ward(sector):
    sector_key = sector.lower()
    for sep in [' ', '_']:
        fname = f"{sector_key.replace(' ', sep)}_l2.csv"
        fpath = os.path.join(L2_DIR, fname)
        if os.path.exists(fpath):
            break
    else:
        return None, None

    df = pd.read_csv(fpath).dropna(subset=['L2_Norm'])
    df = df[df['Fund'] != 'SPY']
    if df.empty:
        return None, None

    X = df[['L2_Norm']].values
    funds = df['Fund'].values
    years = df['Year'].values

    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X) + 1
    df_kmeans = pd.DataFrame({'Fund': funds, 'Year': years, 'KMeans': kmeans_labels})

    ward = AgglomerativeClustering(n_clusters=10, linkage='ward')
    ward_labels = ward.fit_predict(X) + 1
    df_ward = pd.DataFrame({'Fund': funds, 'Year': years, 'Ward': ward_labels})

    return df_kmeans, df_ward

# -----------------------------
# Main analysis
# -----------------------------
pairs = {
    'Utilities': ('PRUAX', 'MMUFX'),
    'Real Estate': ('RPFRX', 'DPREX')
}

for sector, (fund1, fund2) in pairs.items():
    print(f"\n{'='*70}")
    print(f"SECTOR: {sector.upper()}")
    print(f"Comparing: {fund1} vs {fund2}")
    print('='*70)

    # --- Get decile clusters (DataFrame) ---
    df_decile = get_decile_clusters(sector)
    if df_decile is None:
        print(f"  Decile file not found for {sector}")
        continue

    # --- Get KMeans & Ward clusters (DataFrames) ---
    df_kmeans, df_ward = run_kmeans_and_ward(sector)
    if df_kmeans is None:
        print(f"  Could not run KMeans/Ward for {sector}")
        continue

    # Merge all three
    df = df_decile.merge(df_kmeans, on=['Fund', 'Year']).merge(df_ward, on=['Fund', 'Year'])
    df_comp = df[df['Fund'].isin([fund1, fund2])].sort_values(['Year', 'Fund'])

    print("\nYear | {:<4} (D,K,W) | {:<4} (D,K,W) | Diff: D,K,W".format(fund1, fund2))
    print('-'*70)

    # Pivot for easy access
    pivot_dec = df_comp.pivot(index='Year', columns='Fund', values='Decile')
    pivot_km  = df_comp.pivot(index='Year', columns='Fund', values='KMeans')
    pivot_ward= df_comp.pivot(index='Year', columns='Fund', values='Ward')

    for yr in pivot_dec.index:
        d1 = int(pivot_dec.loc[yr, fund1])
        d2 = int(pivot_dec.loc[yr, fund2])
        k1 = int(pivot_km.loc[yr, fund1])
        k2 = int(pivot_km.loc[yr, fund2])
        w1 = int(pivot_ward.loc[yr, fund1])
        w2 = int(pivot_ward.loc[yr, fund2])
        diff_dec = abs(d1 - d2)
        diff_km  = abs(k1 - k2)
        diff_ward= abs(w1 - w2)
        marker = '***' if (diff_dec > 3 or diff_km > 3 or diff_ward > 3) else ''
        print(f"{yr:4d} | {fund1}: {d1},{k1},{w1} | {fund2}: {d2},{k2},{w2} | D={diff_dec}, K={diff_km}, W={diff_ward} {marker}")

    print('\n' + '-'*70)
    print("SUMMARY STATISTICS:")
    print(f"  Decile Hamming distance:    {(pivot_dec[fund1] != pivot_dec[fund2]).sum()} years")
    print(f"  KMeans Hamming distance:    {(pivot_km[fund1] != pivot_km[fund2]).sum()} years")
    print(f"  Ward Hamming distance:      {(pivot_ward[fund1] != pivot_ward[fund2]).sum()} years")
    print(f"  Decile Weighted Hamming:    {abs(pivot_dec[fund1] - pivot_dec[fund2]).sum():.1f}")
    print(f"  KMeans Weighted Hamming:    {abs(pivot_km[fund1] - pivot_km[fund2]).sum():.1f}")
    print(f"  Ward Weighted Hamming:      {abs(pivot_ward[fund1] - pivot_ward[fund2]).sum():.1f}")

    out_path = os.path.join(OUT_DIR, f"{sector}_cluster_comparison.csv")
    df_comp.to_csv(out_path, index=False)
    print(f"\n✓ Full table saved to: {out_path}")

print("\n" + '='*70)
print("Done.")