import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# -----------------------------
# OUTPUT
# -----------------------------
out_dir = "../plots/hammingdistance/intersector"
os.makedirs(out_dir, exist_ok=True)

# Updated file paths to match current data
sector_files = {
    'Technology': "../clusters/technology_quantiles_global.csv",
    'Healthcare': "../clusters/healthcare_quantiles_global.csv",
    'Utilities': "../clusters/utilities_quantiles_global.csv",
    'Energy': "../clusters/energy_quantiles_global.csv",
    'Real Estate': "../clusters/real_estate_quantiles_global.csv"
}

# Updated colors (more distinct)
sector_colors = {
    'Technology': '#1f77b4',   # Blue
    'Healthcare': '#2ca02c',   # Green
    'Utilities': '#ff7f0e',    # Orange
    'Energy': '#d62728',       # Red
    'Real Estate': '#9467bd'   # Purple
}

# -----------------------------
# LOAD + MERGE ALL SECTORS
# -----------------------------
dfs = []
for s, f in sector_files.items():
    if not os.path.exists(f):
        print(f"File not found: {f}")
        continue
    d = pd.read_csv(f)
    d = d.rename(columns={'Decile_Cluster': 'Quantile'})
    d["Sector"] = s
    # Remove SPY if present
    d = d[d['Fund'] != 'SPY']
    dfs.append(d)

df = pd.concat(dfs, ignore_index=True)

# -----------------------------
# TRAJECTORY: Sector-Year matrix using mode
# -----------------------------
def mode_cluster(x):
    return x.mode().iloc[0] if not x.mode().empty else np.nan

yearly = df.groupby(["Year", "Sector"])["Quantile"].agg(mode_cluster).reset_index()
traj = yearly.pivot(index="Year", columns="Sector", values="Quantile").sort_index()

sectors = traj.columns
years = traj.index

print(f"Sectors: {list(sectors)}")
print(f"Years: {len(years)}")

# -----------------------------
# AGGREGATE INTER-SECTOR HAMMING PER SECTOR
# -----------------------------
plt.figure(figsize=(14, 7))

for s in sectors:
    vals = []
    for y in years:
        total = 0
        count = 0
        for other in sectors:
            if other == s:
                continue
            q1 = traj.loc[y, s]
            q2 = traj.loc[y, other]
            if pd.isna(q1) or pd.isna(q2):
                continue
            total += (q1 != q2)
            count += 1
        vals.append(total if count > 0 else np.nan)
    
    plt.plot(years, vals, marker='s', linewidth=2, markersize=5, 
             label=s, color=sector_colors.get(s, 'gray'))

plt.title("Inter-Sector Hamming Divergence\n(Number of sectors with different modal cluster)", fontsize=12)
plt.xlabel("Year", fontsize=10)
plt.ylabel("Number of Sector Pairs with Different Clusters", fontsize=10)
plt.grid(alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "intersector_hamming_pairwise.jpg"), dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✓ Saved: intersector_hamming_pairwise.jpg")