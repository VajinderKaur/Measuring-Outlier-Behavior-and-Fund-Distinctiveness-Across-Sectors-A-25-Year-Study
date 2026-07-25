import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("../plots/trajectory", exist_ok=True)

sector_files = {
    'Technology': "../clusters/technology_quantiles_global.csv",
    'Healthcare': "../clusters/healthcare_quantiles_global.csv",
    'Utilities': "../clusters/utilities_quantiles_global.csv",
    'Energy': "../clusters/energy_quantiles_global.csv",
    'Real Estate': "../clusters/real_estate_quantiles_global.csv"
}

colors = [
    '#0000FF', '#8A2BE2', '#000000', '#FF0000', '#008000',
    '#808080', '#FFA500', '#FF00FF', '#800000', '#00CED1',
    '#00FF7F'
]

def plot_sector(sector_name, file):
    df = pd.read_csv(file)
    
    # Remove SPY from the plot (but keep it for reference line)
    df = df[df['Fund'] != 'SPY']
    
    print(f"  Funds in {sector_name}: {df['Fund'].unique().tolist()}")
    
    funds = df["Fund"].unique()
    color_map = {f: colors[i % len(colors)] for i, f in enumerate(funds)}

    plt.figure(figsize=(14, 7))
    
    # Add S&P 500 baseline reference line at C1
    plt.axhline(y=1, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='S&P 500 Benchmark (C1)')

    for fund, g in df.groupby("Fund"):
        g = g.sort_values("Year")
        plt.plot(g["Year"], g["Decile_Cluster"],
                 marker='o', 
                 markersize=4,
                 label=fund,
                 color=color_map[fund], 
                 linewidth=1.5,
                 alpha=0.8)

    plt.xticks(range(1999, 2026, 2), rotation=45)
    plt.yticks(range(1, 11))
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Decile Cluster (C1 = closest to benchmark, C10 = farthest)", fontsize=12)
    plt.title(f"{sector_name} Sector: Fund Trajectories", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(alpha=0.3)
    plt.ylim(0.5, 10.5)

    plt.tight_layout()
    plt.savefig(f"../plots/trajectory/{sector_name}_trajectories.jpg", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {sector_name}_trajectories.jpg")

# Run for all sectors
for s, f in sector_files.items():
    print(f"\nProcessing {s}...")
    plot_sector(s, f)
    
print("\n✓ All trajectory plots saved to ../plots/trajectory/")