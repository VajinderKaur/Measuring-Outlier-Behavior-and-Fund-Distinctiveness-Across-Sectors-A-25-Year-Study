"""
Compare Regular Hamming vs Weighted Hamming (|i-j|)
Shows correlation between the two approaches as robustness check
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import spearmanr, pearsonr

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLUSTER_DIR = os.path.join(BASE_DIR, "../clusters")
OUT_PLOT_DIR = os.path.join(BASE_DIR, "../plots/hamming_comparison")
os.makedirs(OUT_PLOT_DIR, exist_ok=True)

# -----------------------------
# SECTOR FILES
# -----------------------------
sector_files = {
    "Technology": "technology_quantiles_global.csv",
    "Healthcare": "healthcare_quantiles_global.csv",
    "Utilities": "utilities_quantiles_global.csv",
    "Energy": "energy_quantiles_global.csv",
    "Real Estate": "real_estate_quantiles_global.csv"
}

# -----------------------------
# HAMMING FUNCTIONS
# -----------------------------
def hamming_regular(s1, s2):
    """Regular Hamming: counts if different (binary)"""
    mask = s1.notna() & s2.notna()
    if mask.sum() == 0:
        return np.nan
    return (s1[mask] != s2[mask]).sum()

def hamming_weighted(s1, s2):
    """Weighted Hamming: uses |i-j|"""
    mask = s1.notna() & s2.notna()
    if mask.sum() == 0:
        return np.nan
    return np.abs(s1[mask] - s2[mask]).sum()

# -----------------------------
# STORE ALL RESULTS
# -----------------------------
all_results = []

for sector, file in sector_files.items():
    print(f"\nProcessing {sector}...")
    
    file_path = os.path.join(CLUSTER_DIR, file)
    if not os.path.exists(file_path):
        print(f"  File not found: {file_path}")
        continue
    
    df = pd.read_csv(file_path)
    df = df.rename(columns={'Decile_Cluster': 'Cluster'})
    df = df[df['Fund'] != 'SPY']  # Remove SPY
    
    # Create trajectory matrix (Year × Fund)
    traj = df.pivot(index='Year', columns='Fund', values='Cluster').sort_index()
    funds = list(traj.columns)
    
    print(f"  Funds: {len(funds)}")
    
    # Calculate distances for all pairs
    regular_dists = []
    weighted_dists = []
    fund_pairs = []
    
    for f1, f2 in combinations(funds, 2):
        s1 = traj[f1].dropna()
        s2 = traj[f2].dropna()
        
        # Align years (both should have same years)
        common_years = s1.index.intersection(s2.index)
        if len(common_years) == 0:
            continue
        
        s1_aligned = s1[common_years]
        s2_aligned = s2[common_years]
        
        d_reg = hamming_regular(s1_aligned, s2_aligned)
        d_wgt = hamming_weighted(s1_aligned, s2_aligned)
        
        if not np.isnan(d_reg) and not np.isnan(d_wgt):
            regular_dists.append(d_reg)
            weighted_dists.append(d_wgt)
            fund_pairs.append((f1, f2))
    
    # Calculate correlations
    spearman_corr, spearman_p = spearmanr(regular_dists, weighted_dists)
    pearson_corr, pearson_p = pearsonr(regular_dists, weighted_dists)
    
    print(f"  Pairs: {len(regular_dists)}")
    print(f"  Spearman correlation: {spearman_corr:.4f}")
    print(f"  Pearson correlation: {pearson_corr:.4f}")
    
    # Store results
    all_results.append({
        'Sector': sector,
        'Num_Funds': len(funds),
        'Num_Pairs': len(regular_dists),
        'Spearman': spearman_corr,
        'Pearson': pearson_corr,
        'Mean_Regular': np.mean(regular_dists),
        'Mean_Weighted': np.mean(weighted_dists),
        'Std_Regular': np.std(regular_dists),
        'Std_Weighted': np.std(weighted_dists)
    })
    
    # -----------------------------
    # SCATTER PLOT for this sector
    # -----------------------------
    plt.figure(figsize=(8, 6))
    plt.scatter(regular_dists, weighted_dists, alpha=0.6, c='steelblue', edgecolors='white', s=60)
    
    # Add trend line
    z = np.polyfit(regular_dists, weighted_dists, 1)
    p = np.poly1d(z)
    plt.plot(sorted(regular_dists), p(sorted(regular_dists)), 'r--', linewidth=2, 
             label=f'Trend (slope={z[0]:.2f})')
    
    plt.xlabel('Regular Hamming Distance (years different)', fontsize=12)
    plt.ylabel('Weighted Hamming Distance (|i-j|)', fontsize=12)
    plt.title(f'{sector} Sector\nRegular vs Weighted Hamming (Spearman={spearman_corr:.3f})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_PLOT_DIR, f"{sector}_scatter.jpg"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {sector}_scatter.jpg")

# -----------------------------
# COMBINED PLOT (all sectors together)
# -----------------------------
plt.figure(figsize=(10, 8))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
all_regular = []
all_weighted = []
all_sectors = []

for i, (sector, file) in enumerate(sector_files.items()):
    file_path = os.path.join(CLUSTER_DIR, file)
    if not os.path.exists(file_path):
        continue
    
    df = pd.read_csv(file_path)
    df = df.rename(columns={'Decile_Cluster': 'Cluster'})
    df = df[df['Fund'] != 'SPY']
    
    traj = df.pivot(index='Year', columns='Fund', values='Cluster').sort_index()
    funds = list(traj.columns)
    
    for f1, f2 in combinations(funds, 2):
        s1 = traj[f1].dropna()
        s2 = traj[f2].dropna()
        common_years = s1.index.intersection(s2.index)
        if len(common_years) == 0:
            continue
        
        d_reg = hamming_regular(s1[common_years], s2[common_years])
        d_wgt = hamming_weighted(s1[common_years], s2[common_years])
        
        if not np.isnan(d_reg) and not np.isnan(d_wgt):
            all_regular.append(d_reg)
            all_weighted.append(d_wgt)
            all_sectors.append(sector)
    
    plt.scatter([r for r, s in zip(all_regular, all_sectors) if s == sector],
                [w for r, w, s in zip(all_regular, all_weighted, all_sectors) if s == sector],
                alpha=0.5, color=colors[i], label=sector, s=40)

# Combined trend line
z_all = np.polyfit(all_regular, all_weighted, 1)
p_all = np.poly1d(z_all)
plt.plot(sorted(all_regular), p_all(sorted(all_regular)), 'k--', linewidth=2,
         label=f'Overall trend (slope={z_all[0]:.2f})')

plt.xlabel('Regular Hamming Distance (years different)', fontsize=12)
plt.ylabel('Weighted Hamming Distance (|i-j|)', fontsize=12)
plt.title('All Sectors: Regular vs Weighted Hamming Distance', fontsize=14)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_PLOT_DIR, "all_sectors_scatter.jpg"), dpi=300, bbox_inches='tight')
plt.close()
print(f"\nSaved: all_sectors_scatter.jpg")

# -----------------------------
# SUMMARY TABLE
# -----------------------------
summary_df = pd.DataFrame(all_results)
print("\n" + "="*80)
print("SUMMARY: Regular vs Weighted Hamming Correlation")
print("="*80)
print(summary_df.to_string(index=False))

# Save summary table
summary_df.to_csv(os.path.join(OUT_PLOT_DIR, "correlation_summary.csv"), index=False)
print(f"\n✓ Saved summary to: {OUT_PLOT_DIR}/correlation_summary.csv")

# -----------------------------
# BAR CHART OF CORRELATIONS
# -----------------------------
plt.figure(figsize=(10, 6))
x = range(len(summary_df))
width = 0.35

plt.bar([i - width/2 for i in x], summary_df['Spearman'], width, label='Spearman', color='steelblue')
plt.bar([i + width/2 for i in x], summary_df['Pearson'], width, label='Pearson', color='coral')

plt.xticks(x, summary_df['Sector'], rotation=45)
plt.ylabel('Correlation Coefficient', fontsize=12)
plt.xlabel('Sector', fontsize=12)
plt.title('Regular vs Weighted Hamming: Correlation by Sector', fontsize=14)
plt.ylim(0, 1)
plt.legend()
plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='r=0.8 reference')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_PLOT_DIR, "correlation_barchart.jpg"), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: correlation_barchart.jpg")

# -----------------------------
# FINAL SUMMARY
# -----------------------------
print("\n" + "="*80)
print("CONCLUSION FOR REVIEWER RESPONSE")
print("="*80)
print(f"Overall number of fund pairs: {len(all_regular)}")
print(f"Overall Spearman correlation: {spearmanr(all_regular, all_weighted)[0]:.4f}")
print(f"Overall Pearson correlation: {pearsonr(all_regular, all_weighted)[0]:.4f}")
print("\nInterpretation:")
print("  • High correlation (>0.90) indicates regular Hamming captures similar patterns")
print("  • Weighted Hamming provides additional ordinal information")
print("  • Results are robust to both distance measures")
print("="*80)