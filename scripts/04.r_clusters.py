import os
import glob
import pandas as pd
import numpy as np

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

l2_path = os.path.join(BASE_DIR, "l2")
robustness_path = os.path.join(BASE_DIR, "robustness_clusters")

os.makedirs(robustness_path, exist_ok=True)

# ---------------------------
# Define k values to test
# ---------------------------
K_VALUES = [5, 8, 9, 11, 12]

# ---------------------------
# Step 1: Load L2 norms from ALL sectors
# ---------------------------
print("="*60)
print("LOADING L2 NORMS FROM ALL SECTORS")
print("="*60)

all_l2 = []
for filepath in glob.glob(os.path.join(l2_path, "*_l2.csv")):
    sector = os.path.basename(filepath).replace("_l2.csv", "")
    print(f"  Loading {sector}...")
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['L2_Norm'])
    all_l2.append(df)

combined_df = pd.concat(all_l2, ignore_index=True)
print(f"\nTotal fund-year observations: {len(combined_df)}")

# ---------------------------
# Step 2: Create clusters for each k value
# ---------------------------
print("\n" + "="*60)
print("CREATING ROBUSTNESS CLUSTERS")
print("="*60)

# Store summary data
summary_data = []

for k in K_VALUES:
    print(f"\n--- Processing k={k} ---")
    
    # Create k quantiles
    cluster_labels = range(1, k+1)
    
    try:
        # First attempt: direct qcut
        combined_df[f'Cluster_k{k}'] = pd.qcut(
            combined_df['L2_Norm'],
            q=k,
            labels=cluster_labels,
            duplicates='drop'
        )
        
        actual_clusters = combined_df[f'Cluster_k{k}'].nunique()
        if actual_clusters < k:
            print(f"  WARNING: Only {actual_clusters} clusters created (requested {k})")
            # Fallback: rank-based approach
            combined_df[f'Cluster_k{k}'] = pd.qcut(
                combined_df['L2_Norm'].rank(method='first'),
                q=k,
                labels=cluster_labels
            )
            actual_clusters = combined_df[f'Cluster_k{k}'].nunique()
            print(f"  Rank-based approach: {actual_clusters} clusters created")
        
        # Show boundaries
        print(f"\nCluster boundaries (k={k}):")
        for i in range(1, actual_clusters + 1):
            cluster_data = combined_df[combined_df[f'Cluster_k{k}'] == i]
            if len(cluster_data) > 0:
                min_val = cluster_data['L2_Norm'].min()
                max_val = cluster_data['L2_Norm'].max()
                print(f"  C{i}: {min_val:.4f} - {max_val:.4f} (n={len(cluster_data)})")
        
        # ---------------------------
        # SAVE FILES (matching original structure)
        # ---------------------------
        
        # 1. Save combined file (like all_funds_decile_clusters.csv)
        combined_out = os.path.join(robustness_path, f"all_funds_clusters_k{k}.csv")
        combined_df.to_csv(combined_out, index=False)
        print(f"✓ Saved combined: {combined_out}")
        
        # 2. Save per-sector files (like technology_quantiles_global.csv)
        for sector in combined_df['Sector'].unique():
            sector_df = combined_df[combined_df['Sector'] == sector]
            # Use same naming convention as original: sector_quantiles_global.csv
            # But with k in the filename to avoid overwriting
            sector_filename = f"{sector.lower()}_quantiles_global_k{k}.csv"
            sector_out = os.path.join(robustness_path, sector_filename)
            sector_df.to_csv(sector_out, index=False)
            print(f"✓ Saved sector: {sector_out}")
        
        # 3. Summary stats for this k
        for sector in combined_df['Sector'].unique():
            sector_df = combined_df[combined_df['Sector'] == sector]
            counts = sector_df[f'Cluster_k{k}'].value_counts().sort_index()
            row_data = {'k': k, 'Sector': sector}
            for i in range(1, actual_clusters + 1):
                row_data[f'C{i}'] = counts.get(i, 0)
            summary_data.append(row_data)
            
    except Exception as e:
        print(f"  ERROR with k={k}: {e}")
        continue

# ---------------------------
# Step 3: Create summary table
# ---------------------------
print("\n" + "="*60)
print("SUMMARY: CLUSTER DISTRIBUTIONS BY SECTOR")
print("="*60)

if summary_data:
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.fillna(0)
    
    for k in K_VALUES:
        print(f"\n--- K={k} ---")
        k_data = summary_df[summary_df['k'] == k]
        if not k_data.empty:
            # Convert to percentages
            k_data_numeric = k_data.set_index('Sector')
            total_cols = [col for col in k_data_numeric.columns if col.startswith('C')]
            k_data_numeric['Total'] = k_data_numeric[total_cols].sum(axis=1)
            for col in total_cols:
                k_data_numeric[col] = (k_data_numeric[col] / k_data_numeric['Total']) * 100
            print(k_data_numeric[total_cols].round(1))
    
    summary_out = os.path.join(robustness_path, "cluster_summary_all_k.csv")
    summary_df.to_csv(summary_out, index=False)
    print(f"\n✓ Summary saved to: {summary_out}")

print("\n" + "="*60)
print("FILE STRUCTURE CREATED:")
print("="*60)
print(f"robustness_clusters/")
print(f"├── all_funds_clusters_k5.csv")
print(f"├── all_funds_clusters_k8.csv")
print(f"├── all_funds_clusters_k9.csv")
print(f"├── all_funds_clusters_k11.csv")
print(f"├── all_funds_clusters_k12.csv")
print(f"├── technology_quantiles_global_k5.csv")
print(f"├── technology_quantiles_global_k8.csv")
print(f"├── ... (all sectors for each k)")
print(f"└── cluster_summary_all_k.csv")
print("="*60)