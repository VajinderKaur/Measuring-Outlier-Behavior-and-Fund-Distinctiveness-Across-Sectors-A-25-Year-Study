import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

os.makedirs("plots/hammingdistance/sectors", exist_ok=True)

def hamming(s1, s2):
    mask = s1.notna() & s2.notna()
    if mask.sum() == 0:
        return np.nan
    return (s1[mask] != s2[mask]).mean()

sector_files = {
    'Technology': "clusters/tech_quantiles_byyear.csv",
    'Healthcare': "clusters/hc_quantiles_byyear.csv",
    'Utilities': "clusters/util_quantiles_byyear.csv",
    'Energy': "clusters/energy_quantiles_byyear.csv",
    'Real Estate': "clusters/re_quantiles_byyear.csv"
}

for sector, file in sector_files.items():

    df = pd.read_csv(file)
    years = sorted(df["Year"].unique())

    trajectories = df.pivot(index="Year", columns="Fund", values="Quantile").sort_index()

    funds = trajectories.columns

    plt.figure(figsize=(12,6))

    for f in funds:
        vals = []

        for y in years:
            fund_q = trajectories.loc[y, f]
            others = trajectories.loc[y].drop(f)

            dists = [(fund_q != o) for o in others.dropna()]
            vals.append(np.mean(dists) if len(dists) > 0 else np.nan)

        plt.plot(years, vals, marker='o', linewidth=1, label=f)

    plt.title(f"{sector} - Within-Sector Hamming")
    plt.xlabel("Year")
    plt.ylabel("Avg Hamming vs Other Funds")
    plt.ylim(0,1.1)
    plt.grid(alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')

    plt.tight_layout()
    plt.savefig(f"plots/hammingdistance/sectors/{sector}.jpg", dpi=1200)
    plt.close()