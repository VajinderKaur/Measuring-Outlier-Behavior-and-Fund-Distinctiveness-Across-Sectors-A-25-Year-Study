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

# ---------------------------
# Load all precomputed L2 files
# ---------------------------
all_dfs = []

for file in glob.glob(os.path.join(l2_path, "*_l2.csv")):
    df = pd.read_csv(file)

    # safety check
    required_cols = {'Fund', 'Year', 'Sector', 'L2_Distance'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing columns in {file}")

    df = df.dropna(subset=['L2_Distance'])
    all_dfs.append(df)

final_df = pd.concat(all_dfs, ignore_index=True)

# ---------------------------
# Sector order (stable)
# ---------------------------
sector_order = sorted(final_df['Sector'].unique())

# ---------------------------
# Y-axis consistency (important for comparison)
# ---------------------------
y_min = final_df['L2_Distance'].quantile(0.01)
y_max = final_df['L2_Distance'].quantile(0.99)

# ---------------------------
# Plot
# ---------------------------
plt.figure(figsize=(11, 6))

sns.boxplot(
    data=final_df,
    x='Sector',
    y='L2_Distance',
    order=sector_order,
    width=0.6,            # tighter, cleaner boxes
    fliersize=3           # smaller outlier dots
)

# ---------------------------
# Better y-axis padding
# ---------------------------
y_min = final_df['L2_Distance'].quantile(0.01)
y_max = final_df['L2_Distance'].quantile(0.995)

# ADD PADDING (this is key fix)
padding = (y_max - y_min) * 0.08

plt.ylim(y_min - padding, y_max + padding)

# ---------------------------
# Styling improvements
# ---------------------------
plt.title("Sector-wise Distribution of Fund-Year L2 Residuals", fontsize=14)
plt.xlabel("Sector", fontsize=12)
plt.ylabel("L2 Distance (RMS Residual)", fontsize=12)

plt.xticks(rotation=20)

plt.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()

# ---------------------------
# Save figure
# ---------------------------
out_file = os.path.join(plot_path, "sector_boxplots_L2.png")
plt.savefig(out_file, dpi=300)
plt.close()

print(f"Saved boxplot: {out_file}")