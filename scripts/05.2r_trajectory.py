import os
import pandas as pd
import matplotlib.pyplot as plt

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
robustness_path = os.path.join(BASE_DIR, "robustness_clusters")
plots_path = os.path.join(BASE_DIR, "plots", "robustness_trajectories")

os.makedirs(plots_path, exist_ok=True)

# Define k values and sectors
K_VALUES = [5, 8, 9, 11, 12]
SECTORS = ['Technology', 'Healthcare', 'Utilities', 'Energy', 'Real Estate']

# Color palette
colors = [
    '#0000FF', '#8A2BE2', '#000000', '#FF0000', '#008000',
    '#808080', '#FFA500', '#FF00FF', '#800000', '#00CED1',
    '#00FF7F', '#FF1493', '#4B0082', '#00FF00'
]

def plot_sector_robustness(sector, k):
    """
    Plot trajectories for a specific sector and k value
    """
    # Construct filename matching original pattern
    sector_key = sector.lower()
    filename = f"{sector_key}_quantiles_global_k{k}.csv"
    filepath = os.path.join(robustness_path, filename)
    
    if not os.path.exists(filepath):
        print(f"  File not found: {filepath}")
        return
    
    # Load data
    df = pd.read_csv(filepath)
    
    # Remove SPY from plot
    df = df[df['Fund'] != 'SPY']
    
    if df.empty:
        print(f"  No funds (excluding SPY) for {sector}, k={k}")
        return
    
    funds = df["Fund"].unique()
    color_map = {f: colors[i % len(colors)] for i, f in enumerate(funds)}
    
    cluster_col = f'Cluster_k{k}'
    
    plt.figure(figsize=(14, 7))
    
    # Add benchmark reference line at C1
    plt.axhline(y=1, color='black', linestyle='--', linewidth=1.5, alpha=0.7, 
                label='S&P 500 Benchmark (C1)')
    
    for fund, g in df.groupby("Fund"):
        g = g.sort_values("Year")
        plt.plot(g["Year"], g[cluster_col],
                 marker='o', 
                 markersize=4,
                 label=fund,
                 color=color_map[fund], 
                 linewidth=1.5,
                 alpha=0.8)
    
    # Set y-axis ticks based on k
    y_ticks = range(1, k+1)
    y_max = k + 0.5
    
    plt.xticks(range(1999, 2026, 2), rotation=45)
    plt.yticks(y_ticks)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel(f"Cluster (C1 = closest, C{k} = farthest)", fontsize=12)
    plt.title(f"{sector} Sector: Fund Trajectories (k={k} clusters)", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(alpha=0.3)
    plt.ylim(0.5, y_max)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"{sector_key}_trajectories_k{k}.jpg"
    plot_path = os.path.join(plots_path, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {plot_filename}")

# ---------------------------
# Main execution
# ---------------------------
print("="*60)
print("CREATING ROBUSTNESS TRAJECTORY PLOTS")
print("="*60)

for k in K_VALUES:
    print(f"\n--- Processing k={k} ---")
    for sector in SECTORS:
        plot_sector_robustness(sector, k)

print("\n" + "="*60)
print(f"✓ All robustness plots saved to: {plots_path}")
print(f"✓ Created plots for k values: {K_VALUES}")
print("="*60)