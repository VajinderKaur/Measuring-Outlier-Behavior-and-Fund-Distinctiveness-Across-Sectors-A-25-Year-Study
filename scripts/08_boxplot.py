import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

l2_path = os.path.join(BASE_DIR, "l2")
plot_path = os.path.join(BASE_DIR, "plots", "boxplots")

os.makedirs(plot_path, exist_ok=True)

# Updated sector colors with correct names
SECTOR_COLORS = {
    "energy": "#1f77b4",       # blue
    "technology": "#ff7f0e",   # orange
    "healthcare": "#2ca02c",   # green
    "utilities": "#d62728",    # red
    "real estate": "#9467bd"   # purple
}

# ---------------------------
# Load all precomputed L2 files
# ---------------------------
all_dfs = []

for file in glob.glob(os.path.join(l2_path, "*_l2.csv")):
    df = pd.read_csv(file)
    
    # Use correct column name: L2_Norm
    required_cols = {'Fund', 'Year', 'Sector', 'L2_Norm'}
    if not required_cols.issubset(df.columns):
        print(f"Warning: {file} missing columns. Found: {df.columns.tolist()}")
        continue
    
    df = df.dropna(subset=['L2_Norm'])
    all_dfs.append(df)

final_df = pd.concat(all_dfs, ignore_index=True)

# ---------------------------
# Sector order (stable)
# ---------------------------
sector_order = ["energy", "technology", "healthcare", "utilities", "real estate"]

# ---------------------------
# Plot
# ---------------------------
plt.figure(figsize=(11, 6))

sns.boxplot(
    data=final_df,
    x='Sector',
    y='L2_Norm',
    order=sector_order,
    hue='Sector',
    palette=SECTOR_COLORS,
    width=0.6,
    fliersize=3,
    dodge=False,
    legend=False
)

# Better y-axis padding
y_min = final_df['L2_Norm'].quantile(0.01)
y_max = final_df['L2_Norm'].quantile(0.995)
padding = (y_max - y_min) * 0.08
plt.ylim(y_min - padding, y_max + padding)

# Styling
plt.title("Sector-wise Distribution of Fund-Year L2 Residual Norms", fontsize=14)
plt.xlabel("Sector", fontsize=12)
plt.ylabel("L2 Norm (Euclidean length of residual vector)", fontsize=12)

plt.xticks(rotation=20)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()

# Save figure
out_file = os.path.join(plot_path, "sector_boxplots_L2.png")
plt.savefig(out_file, dpi=300)
plt.close()

print(f"✓ Saved boxplot: {out_file}")