import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("../plots/hammingdistance/sectors", exist_ok=True)

# Updated file paths
sector_files = {
    'Technology': "../clusters/technology_quantiles_global.csv",
    'Healthcare': "../clusters/healthcare_quantiles_global.csv",
    'Utilities': "../clusters/utilities_quantiles_global.csv",
    'Energy': "../clusters/energy_quantiles_global.csv",
    'Real Estate': "../clusters/real_estate_quantiles_global.csv"
}

for sector, file in sector_files.items():
    
    if not os.path.exists(file):
        print(f"File not found: {file}")
        continue
    
    df = pd.read_csv(file)
    df = df.rename(columns={'Decile_Cluster': 'Quantile'})
    
    # Remove SPY if present
    df = df[df['Fund'] != 'SPY']
    
    df["Quantile"] = pd.to_numeric(df["Quantile"], errors="coerce")
    years = sorted(df["Year"].unique())
    
    traj = df.pivot(index="Year", columns="Fund", values="Quantile").sort_index()
    funds = traj.columns
    
    plt.figure(figsize=(14, 8))
    
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
            diffs = (others != fund_q).dropna()
            vals.append(diffs.sum())
        
        plt.plot(years, vals, marker='o', linewidth=1, label=f, alpha=0.7)
    
    plt.title(f"{sector} Sector — Cross-Sectional Dispersion\n(Number of peer funds in different deciles)", fontsize=12)
    plt.xlabel("Year", fontsize=10)
    plt.ylabel("Number of Funds in Different Deciles", fontsize=10)
    plt.grid(alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(f"../plots/hammingdistance/sectors/{sector}_dispersion.jpg", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {sector}_dispersion.jpg")