import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("plots/trajectory", exist_ok=True)

sector_files = {
    'Technology': "clusters/tech_quantiles_byyear.csv",
    'Healthcare': "clusters/hc_quantiles_byyear.csv",
    'Utilities': "clusters/util_quantiles_byyear.csv",
    'Energy': "clusters/energy_quantiles_byyear.csv",
    'Real Estate': "clusters/re_quantiles_byyear.csv"
}

colors = [
    '#0000FF', '#8A2BE2', '#000000', '#FF0000', '#008000',
    '#808080', '#FFA500', '#FF00FF', '#800000', '#00CED1',
    '#00FF7F'
]

def plot_sector(sector_name, file):
    df = pd.read_csv(file)

    funds = df["Fund"].unique()
    color_map = {f: colors[i % len(colors)] for i, f in enumerate(funds)}

    plt.figure(figsize=(12,6))

    for fund, g in df.groupby("Fund"):
        g = g.sort_values("Year")
        plt.plot(g["Year"], g["Quantile"],
                 marker='o', label=fund,
                 color=color_map[fund], linewidth=1)

    plt.xticks(sorted(df["Year"].unique()), rotation=45)
    plt.yticks(range(1,11))
    plt.xlabel("Year")
    plt.ylabel("L2 Quantile")
    plt.title(f"{sector_name} Fund Trajectories")
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"plots/trajectory/{sector_name}.jpg", dpi=1200)
    plt.close()

for s, f in sector_files.items():
    plot_sector(s, f)