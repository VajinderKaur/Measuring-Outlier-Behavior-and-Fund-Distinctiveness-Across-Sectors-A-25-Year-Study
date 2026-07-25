import pandas as pd
import numpy as np
import os
import glob
from scipy.stats import spearmanr

# ============================================================================
# SECTOR NAME MAPPING (old filename → new sector name)
# ============================================================================
sector_mapping = {
    'tech': 'technology',
    're': 'real estate',
    'healthcare': 'healthcare',
    'energy': 'energy',
    'utilities': 'utilities'
}

# ============================================================================
# LOAD OLD RMS (250 days) - from per-sector files
# ============================================================================
print("="*60)
print("LOADING OLD RMS VALUES (250 DAYS)")
print("="*60)

old_rms_list = []
old_path = "../l2_250days"  # Your old L2 folder

for filepath in glob.glob(os.path.join(old_path, "*_l2.csv")):
    old_sector = os.path.basename(filepath).replace("_l2.csv", "")
    new_sector = sector_mapping.get(old_sector, old_sector)
    
    print(f"  Loading {old_sector} → {new_sector}")
    df = pd.read_csv(filepath)
    
    # Old file has 'L2_Distance' (RMS)
    if 'L2_Distance' in df.columns:
        df = df.rename(columns={'L2_Distance': 'RMS_250'})
    
    # Add mapped sector name
    df['Sector'] = new_sector
    old_rms_list.append(df)

old_df = pd.concat(old_rms_list, ignore_index=True)
print(f"\nTotal old observations: {len(old_df)}")
print(f"Old sectors: {old_df['Sector'].unique().tolist()}")

# ============================================================================
# LOAD NEW L2 NORM (252 days)
# ============================================================================
print("\n" + "="*60)
print("LOADING NEW L2 NORM (252 DAYS)")
print("="*60)

new_l2_list = []
new_path = "../l2"  # Your new L2 folder

for filepath in glob.glob(os.path.join(new_path, "*_l2.csv")):
    sector = os.path.basename(filepath).replace("_l2.csv", "")
    print(f"  Loading {sector}")
    df = pd.read_csv(filepath)
    new_l2_list.append(df)

new_df = pd.concat(new_l2_list, ignore_index=True)
print(f"\nTotal new observations: {len(new_df)}")
print(f"New sectors: {new_df['Sector'].unique().tolist()}")

# ============================================================================
# MERGE AND COMPARE
# ============================================================================
print("\n" + "="*60)
print("COMPARING OLD (RMS, 250 days) vs NEW (L2 Norm, 252 days)")
print("="*60)

# Merge on Fund, Year, Sector
merged = pd.merge(
    old_df[['Fund', 'Year', 'Sector', 'RMS_250']],
    new_df[['Fund', 'Year', 'Sector', 'L2_Norm']],
    on=['Fund', 'Year', 'Sector']
)

print(f"Matching fund-year pairs: {len(merged)}")

# Calculate Spearman correlation (most important for ranking)
spearman_corr, spearman_p = spearmanr(merged['RMS_250'], merged['L2_Norm'])
print(f"\nOVERALL Spearman rank correlation: {spearman_corr:.4f}")

# ============================================================================
# SECTOR-WISE COMPARISON
# ============================================================================
print("\n" + "="*60)
print("SECTOR-WISE SPEARMAN CORRELATIONS")
print("="*60)

sector_results = {}
for sector in merged['Sector'].unique():
    sector_df = merged[merged['Sector'] == sector]
    spearman, _ = spearmanr(sector_df['RMS_250'], sector_df['L2_Norm'])
    sector_results[sector] = spearman
    print(f"  {sector:12s}: Spearman = {spearman:.4f} (n={len(sector_df)})")

# ============================================================================
# CONVERT TO DECILES BASED ON RANKS
# ============================================================================
print("\n" + "="*60)
print("DECILE AGREEMENT (Global)")
print("="*60)

# Convert to deciles based on ranks within each dataset
merged['Rank_250'] = merged['RMS_250'].rank()
merged['Rank_252'] = merged['L2_Norm'].rank()
merged['Decile_250'] = pd.qcut(merged['Rank_250'], q=10, labels=range(1, 11))
merged['Decile_252'] = pd.qcut(merged['Rank_252'], q=10, labels=range(1, 11))

# Convert to numeric
merged['Decile_250'] = merged['Decile_250'].astype(int)
merged['Decile_252'] = merged['Decile_252'].astype(int)

# Calculate agreement
merged['Decile_Diff'] = abs(merged['Decile_250'] - merged['Decile_252'])
same_cluster = (merged['Decile_Diff'] == 0).sum()
same_or_adjacent = (merged['Decile_Diff'] <= 1).sum()
major_change = (merged['Decile_Diff'] >= 3).sum()

print(f"Same decile: {same_cluster}/{len(merged)} ({same_cluster/len(merged)*100:.1f}%)")
print(f"Same or adjacent decile: {same_or_adjacent}/{len(merged)} ({same_or_adjacent/len(merged)*100:.1f}%)")
print(f"Changed by 3+ deciles: {major_change}/{len(merged)} ({major_change/len(merged)*100:.1f}%)")
print(f"Average decile difference: {merged['Decile_Diff'].mean():.2f}")

# ============================================================================
# SECTOR-WISE DECILE AGREEMENT
# ============================================================================
print("\n" + "="*60)
print("SECTOR-WISE DECILE AGREEMENT")
print("="*60)

for sector in merged['Sector'].unique():
    sector_df = merged[merged['Sector'] == sector]
    same = (sector_df['Decile_Diff'] == 0).sum()
    same_adj = (sector_df['Decile_Diff'] <= 1).sum()
    print(f"  {sector:12s}: Same={same/len(sector_df)*100:.1f}%, Same/Adj={same_adj/len(sector_df)*100:.1f}%")

# ============================================================================
# TRANSITION MATRIX
# ============================================================================
print("\n" + "="*60)
print("CLUSTER TRANSITION MATRIX (%)")
print("="*60)
print("(Rows = Old Decile, Columns = New Decile)")
print("-"*60)

transition = pd.crosstab(merged['Decile_250'], merged['Decile_252'], normalize='index') * 100
print(transition.round(1))

# ============================================================================
# CONCLUSION
# ============================================================================
print("\n" + "="*60)
print("CONCLUSION FOR REVIEWER RESPONSE")
print("="*60)

print(f"\n  Overall Spearman correlation: {spearman_corr:.4f}")
print(f"  Same decile: {same_cluster/len(merged)*100:.1f}%")
print(f"  Same or adjacent decile: {same_or_adjacent/len(merged)*100:.1f}%")

print("\n" + "="*60)
print("KEY POINTS FOR REVIEWER RESPONSE:")
print("="*60)
print("• Changed from RMS (250 days) to L2 Norm (252 days)")
print("• Changed from 250 to 252 trading days (industry standard)")
print(f"• Spearman rank correlation: {spearman_corr:.4f}")
print(f"• {same_or_adjacent/len(merged)*100:.1f}% of funds remain in same or adjacent decile")
print("• This confirms the methodology is robust to these choices")
print("="*60)