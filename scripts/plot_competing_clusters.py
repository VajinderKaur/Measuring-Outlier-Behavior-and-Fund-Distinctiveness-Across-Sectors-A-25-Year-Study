"""
plot_competing_clusters.py
Create a figure showing cluster assignments for competing funds
across decile, KMeans, and Ward for Utilities and Real Estate.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IN_DIR = os.path.join(BASE_DIR, "plots", "competing_clusters")
OUT_DIR = os.path.join(BASE_DIR, "plots", "competing_clusters")
os.makedirs(OUT_DIR, exist_ok=True)

# Define sector and fund pairs
pairs = {
    'Utilities': {'file': 'Utilities_cluster_comparison.csv', 'fund1': 'PRUAX', 'fund2': 'MMUFX'},
    'Real Estate': {'file': 'Real Estate_cluster_comparison.csv', 'fund1': 'RPFRX', 'fund2': 'DPREX'}
}

methods = ['Decile', 'KMeans', 'Ward']

# Create a 2x3 grid
fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.25, wspace=0.15)

for row, (sector, info) in enumerate(pairs.items()):
    filepath = os.path.join(IN_DIR, info['file'])
    df = pd.read_csv(filepath)
    fund1 = info['fund1']
    fund2 = info['fund2']
    
    # Pivot for easy plotting
    for col, method in enumerate(methods):
        ax = axes[row, col]
        
        # Get the two funds' cluster assignments for this method
        # The CSV has columns: Fund, Year, Decile, KMeans, Ward
        # We'll filter and pivot
        df_method = df[['Year', 'Fund', method]].pivot(index='Year', columns='Fund', values=method)
        
        # Ensure both funds are present
        if fund1 not in df_method.columns or fund2 not in df_method.columns:
            ax.text(0.5, 0.5, 'Data missing', transform=ax.transAxes, ha='center')
            continue
        
        # Plot trajectories
        ax.plot(df_method.index, df_method[fund1], 'o-', label=fund1, color='#1f77b4', linewidth=1.5, markersize=4)
        ax.plot(df_method.index, df_method[fund2], 's-', label=fund2, color='#ff7f0e', linewidth=1.5, markersize=4)
        
        # Highlight years where difference > 3
        diff = (df_method[fund1] - df_method[fund2]).abs()
        highlight_years = diff[diff > 3].index
        for yr in highlight_years:
            ax.axvspan(yr - 0.4, yr + 0.4, alpha=0.2, color='gray', linewidth=0)
        
        # Add vertical grid for readability
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.5, 10.5)
        ax.set_yticks(range(1, 11))
        
        # Labels and titles
        if row == 0:
            ax.set_title(method, fontsize=12, fontweight='bold')
        if col == 0:
            ax.set_ylabel(f'{sector}\nCluster', fontsize=10)
        else:
            ax.set_ylabel('')
        
        # Legend only for the first column (to avoid clutter)
        if col == 0:
            ax.legend(loc='upper left', fontsize=8)
        
        # Add difference annotation (weighted Hamming) in the corner
        weighted_diff = diff.sum()
        ax.text(0.98, 0.05, f'Weighted Diff = {weighted_diff:.0f}', 
                transform=ax.transAxes, ha='right', fontsize=8, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Common x-axis label
fig.text(0.5, 0.02, 'Year', ha='center', fontsize=12)

# Overall title
fig.suptitle('Cluster Assignments for Competing Funds Across Methods', fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(OUT_DIR, 'competing_funds_clusters.jpg'), dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Figure saved to: {os.path.join(OUT_DIR, 'competing_funds_clusters.jpg')}")