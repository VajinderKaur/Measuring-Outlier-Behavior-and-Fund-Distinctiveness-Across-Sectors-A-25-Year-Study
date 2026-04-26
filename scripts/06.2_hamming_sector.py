import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("../plots/hammingdistance/sectors", exist_ok=True)

sector_files = {
    'Technology': "../clusters/tech_quantiles_byyear.csv",
    'Healthcare': "../clusters/healthcare_quantiles_byyear.csv",
    'Utilities': "../clusters/utilities_quantiles_byyear.csv",
    'Energy': "../clusters/energy_quantiles_byyear.csv",
    'Real Estate': "../clusters/re_quantiles_byyear.csv"
}

for sector, file in sector_files.items():

    df = pd.read_csv(file)
    df["Quantile"] = pd.to_numeric(df["Quantile"], errors="coerce")

    years = sorted(df["Year"].unique())

    traj = df.pivot(index="Year", columns="Fund", values="Quantile").sort_index()

    funds = traj.columns

    plt.figure(figsize=(12, 6))

    for f in funds:

        vals = []

        for y in years:

            if y not in traj.index:
                vals.append(np.nan)
                continue

            fund_q = traj.loc[y, f]

            if pd.isna(fund_q):
                vals.append(np.nan)
                continue

            others = traj.loc[y].drop(f)

            # RAW mismatch counts (NOT normalized)
            diffs = (others != fund_q).dropna()

            vals.append(diffs.sum())   # 🔥 KEY CHANGE

        plt.plot(years, vals, marker='o', linewidth=1, label=f)

    plt.title(f"{sector} - Cross-Sectional Regime Dispersion (Raw Counts)")
    plt.xlabel("Year")
    plt.ylabel("Number of Funds in Different Deciles")
    plt.grid(alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(f"../plots/hammingdistance/sectors/{sector}_dispersion.jpg", dpi=300)
    plt.close()