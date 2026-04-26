import os
import pandas as pd
import numpy as np
from scipy.stats import mode

# -----------------------------
# PATHS
# -----------------------------
base_out = "../plots/hammingdistance/sectors"

sector_dirs = [
    "Technology",
    "Healthcare",
    "Utilities",
    "Energy",
    "Real Estate"
]

# -----------------------------
# STORAGE
# -----------------------------
all_stats = []

# -----------------------------
# LOOP OVER SECTORS
# -----------------------------
for sector in sector_dirs:

    print(f"Processing stats: {sector}")

    file_path = os.path.join(base_out, sector, "hamming_matrix.csv")

    if not os.path.exists(file_path):
        print(f"Missing: {file_path}")
        continue

    df = pd.read_csv(file_path, index_col=0)

    # -----------------------------
    # Flatten matrix (upper triangle only)
    # -----------------------------
    values = df.values
    vals = values[np.triu_indices_from(values, k=1)]

    vals = vals[~np.isnan(vals)]

    if len(vals) == 0:
        continue

    # -----------------------------
    # STATS
    # -----------------------------
    q1 = np.percentile(vals, 25)
    q3 = np.percentile(vals, 75)
    iqr = q3 - q1

    stats = {
        "Sector": sector,
        "Mean": np.mean(vals),
        "Median": np.median(vals),
        "Mode": mode(vals, keepdims=False).mode if len(vals) > 0 else np.nan,
        "Std": np.std(vals),
        "Min": np.min(vals),
        "Max": np.max(vals),
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Count": len(vals)
    }

    all_stats.append(stats)

# -----------------------------
# SAVE OUTPUT
# -----------------------------
stats_df = pd.DataFrame(all_stats)
out_file = os.path.join(base_out, "hamming_summary_stats.csv")
stats_df.to_csv(out_file, index=False)

print("\nSaved summary stats to:", out_file)
print(stats_df)