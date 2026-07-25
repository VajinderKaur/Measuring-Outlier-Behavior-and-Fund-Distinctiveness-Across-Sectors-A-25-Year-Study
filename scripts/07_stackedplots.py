import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sector_files = {
    'Technology': "../clusters/technology_quantiles_global.csv",
    'Healthcare': "../clusters/healthcare_quantiles_global.csv",
    'Utilities': "../clusters/utilities_quantiles_global.csv",
    'Energy': "../clusters/energy_quantiles_global.csv",
    'Real Estate': "../clusters/real_estate_quantiles_global.csv"
}

out_base = "../plots/stackedplots"
os.makedirs(out_base, exist_ok=True)

# Map deciles → 5 groups
def map_group(q):
    if q <= 2:
        return "D1-2 (Closest)"
    elif q <= 4:
        return "D3-4"
    elif q <= 6:
        return "D5-6"
    elif q <= 8:
        return "D7-8"
    else:
        return "D9-10 (Farthest)"

# Color scheme (green = close, red = far)
colors = {
    "D1-2 (Closest)": "#2ca02c",
    "D3-4": "#4c9a46",
    "D5-6": "#ff7f0e",
    "D7-8": "#d62728",
    "D9-10 (Farthest)": "#8b0000"
}

for sector, file in sector_files.items():
    
    if not os.path.exists(file):
        print(f"File not found: {file}")
        continue
    
    df = pd.read_csv(file)
    df = df.rename(columns={'Decile_Cluster': 'Quantile'})
    
    # Remove SPY
    df = df[df['Fund'] != 'SPY']
    
    df["Quantile"] = pd.to_numeric(df["Quantile"], errors="coerce")
    df["Group"] = df["Quantile"].apply(map_group)
    
    # Yearly percentage table
    yearly = (
        df.groupby(["Year", "Group"])
          .size()
          .unstack(fill_value=0)
    )
    
    yearly = yearly.div(yearly.sum(axis=1), axis=0) * 100
    yearly = yearly.sort_index()
    
    # Plot
    plt.figure(figsize=(14, 6))
    
    bottom = np.zeros(len(yearly))
    
    for group in yearly.columns:
        plt.bar(yearly.index, yearly[group], bottom=bottom, 
                label=group, color=colors.get(group, 'gray'))
        bottom += yearly[group].values
    
    plt.title(f"{sector} Sector — Decile Composition Over Time\n(1999-2025)", fontsize=12)
    plt.ylabel("Percentage of Funds (%)", fontsize=10)
    plt.xlabel("Year", fontsize=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0, 100)
    plt.tight_layout()
    
    plt.savefig(os.path.join(out_base, f"{sector}_stacked.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {sector}_stacked.png")