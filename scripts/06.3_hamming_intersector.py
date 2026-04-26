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

sector_files = {
    'Technology': "../clusters/tech_quantiles_byyear.csv",
    'Healthcare': "../clusters/healthcare_quantiles_byyear.csv",
    'Utilities': "../clusters/utilities_quantiles_byyear.csv",
    'Energy': "../clusters/energy_quantiles_byyear.csv",
    'Real Estate': "../clusters/re_quantiles_byyear.csv"
}

sector_colors = ['#0000FF', '#000000', '#008000', '#FFA500', '#800000']

# -----------------------------
# LOAD + MERGE ALL SECTORS
# -----------------------------
dfs = []
for s, f in sector_files.items():
    d = pd.read_csv(f)
    d["Sector"] = s
    dfs.append(d)

df = pd.concat(dfs, ignore_index=True)

# -----------------------------
# TRAJECTORY: Sector-Year matrix
# -----------------------------
def mode(x):
    return x.mode().iloc[0] if not x.mode().empty else np.nan

yearly = df.groupby(["Year", "Sector"])["Quantile"].agg(mode).reset_index()

traj = yearly.pivot(index="Year", columns="Sector", values="Quantile").sort_index()

sectors = traj.columns
years = traj.index

# -----------------------------
# PAIRWISE HAMMING OVER TIME
# -----------------------------
def hamming_series(s1, s2):
    mask = s1.notna() & s2.notna()
    return (s1[mask] != s2[mask]).astype(int)

# store results per pair
pair_results = {}

for s1, s2 in combinations(sectors, 2):
    pair_results[(s1, s2)] = hamming_series(traj[s1], traj[s2])

# -----------------------------
# AGGREGATE INTER-SECTOR HAMMING PER SECTOR
# (symmetric, paper-consistent)
# -----------------------------
plt.figure(figsize=(12, 6))

for i, s in enumerate(sectors):

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

    plt.plot(years, vals, marker='s', label=s, color=sector_colors[i])

# -----------------------------
# PLOT FORMATTING
# -----------------------------
plt.title("Inter-Sector Hamming Divergence (Pairwise, Raw Counts)")
plt.xlabel("Year")
plt.ylabel("Pairwise Sector Disagreements")
plt.grid(alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "intersector_hamming_pairwise.jpg"), dpi=300)
plt.close()