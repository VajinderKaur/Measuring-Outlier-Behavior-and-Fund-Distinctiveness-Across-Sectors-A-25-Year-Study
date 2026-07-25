import os
import glob
import pandas as pd

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

l2_path = os.path.join(BASE_DIR, "l2")
output_path = os.path.join(BASE_DIR, "clusters")

os.makedirs(output_path, exist_ok=True)

# ---------------------------
# Step 1: Combine ALL sectors first (as per methodology)
# ---------------------------
print("="*60)
print("LOADING L2 NORMS FROM ALL SECTORS")
print("="*60)

all_l2 = []
for filepath in glob.glob(os.path.join(l2_path, "*_l2.csv")):
    sector = os.path.basename(filepath).replace("_l2.csv", "")
    print(f"  Loading {sector}...")
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['L2_Norm'])  # Note: column name should be 'L2_Norm'
    all_l2.append(df)

# Combine all data
combined_df = pd.concat(all_l2, ignore_index=True)
print(f"\nTotal fund-year observations: {len(combined_df)}")

# ---------------------------
# Step 2: Calculate global deciles (across ALL funds and years)
# ---------------------------
print("\n" + "="*60)
print("CALCULATING GLOBAL DECILES")
print("="*60)

combined_df['Decile_Cluster'] = pd.qcut(
    combined_df['L2_Norm'],
    q=10,
    labels=range(1, 11)
)

# Show decile boundaries
print("\nDecile boundaries (L2_Norm):")
for i in range(1, 11):
    decile_data = combined_df[combined_df['Decile_Cluster'] == i]
    min_val = decile_data['L2_Norm'].min()
    max_val = decile_data['L2_Norm'].max()
    print(f"  C{i}: {min_val:.4f} - {max_val:.4f} (n={len(decile_data)})")

# ---------------------------
# Step 3: Save combined cluster file (for trajectory analysis)
# ---------------------------
combined_out = os.path.join(output_path, "all_funds_decile_clusters.csv")
combined_df.to_csv(combined_out, index=False)
print(f"\n✓ Combined clusters saved to: {combined_out}")

# ---------------------------
# Step 4: Also save per-sector files (for convenience)
# ---------------------------
for sector in combined_df['Sector'].unique():
    sector_df = combined_df[combined_df['Sector'] == sector]
    out_file = os.path.join(output_path, f"{sector}_quantiles_global.csv")
    sector_df.to_csv(out_file, index=False)
    print(f"✓ Saved: {out_file}")

# ---------------------------
# Summary statistics
# ---------------------------
print("\n" + "="*60)
print("CLUSTER DISTRIBUTION BY SECTOR")
print("="*60)

pivot = pd.crosstab(combined_df['Sector'], combined_df['Decile_Cluster'], normalize='index') * 100
print(pivot.round(1))

print("\n" + "="*60)
print("KEY METHODOLOGICAL NOTE:")
print("="*60)
print("• Deciles calculated ACROSS all funds and years (global distribution)")
print("• C1 = smallest 10% of L2 norms (closest to benchmark)")
print("• C10 = largest 10% of L2 norms (farthest from benchmark)")
print("• This matches methodology section 5.3")
print("="*60)