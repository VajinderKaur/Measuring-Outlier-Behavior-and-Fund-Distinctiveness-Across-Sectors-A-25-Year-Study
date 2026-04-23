import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("plots/hammingdistance/intersector", exist_ok=True)

sector_files = {
    'Technology': "clusters/tech_quantiles_byyear.csv",
    'Healthcare': "clusters/hc_quantiles_byyear.csv",
    'Utilities': "clusters/util_quantiles_byyear.csv",
    'Energy': "clusters/energy_quantiles_byyear.csv",
    'Real Estate': "clusters/re_quantiles_byyear.csv"
}

sector_colors = ['#0000FF', '#000000', '#008000', '#FFA500', '#800000']

def mode(series):
    return series.mode().iloc[0] if not series.mode().empty else np.nan

dfs = []
for s, f in sector_files.items():
    d = pd.read_csv(f)
    d["Sector"] = s
    dfs.append(d)

df = pd.concat(dfs)

yearly = df.groupby(["Year","Sector"])["Quantile"].agg(mode).reset_index()
traj = yearly.pivot(index="Year", columns="Sector", values="Quantile").sort_index()

years = traj.index
sectors = traj.columns

plt.figure(figsize=(12,6))

for i, s in enumerate(sectors):
    vals = []

    for y in years:
        q = traj.loc[y, s]
        others = traj.loc[y].drop(s)

        d = (q != others).mean()
        vals.append(d)

    plt.plot(years, vals, marker='s', label=s, color=sector_colors[i])

plt.title("Inter-Sector Hamming Distance")
plt.xlabel("Year")
plt.ylabel("Avg Hamming vs Other Sectors")
plt.ylim(0,1.1)
plt.grid(alpha=0.3)
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')

plt.tight_layout()
plt.savefig("plots/hammingdistance/intersector/intersector.jpg", dpi=1200)
plt.close()