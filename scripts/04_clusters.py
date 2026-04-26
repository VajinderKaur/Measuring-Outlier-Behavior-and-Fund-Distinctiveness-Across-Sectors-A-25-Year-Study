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
# Loop through each sector L2 file
# ---------------------------
for filepath in glob.glob(os.path.join(l2_path, "*_l2.csv")):

    sector = os.path.basename(filepath).replace("_l2.csv", "")
    print(f"Processing sector: {sector}")

    # ---------------------------
    # Load precomputed L2
    # ---------------------------
    df = pd.read_csv(filepath)

    # IMPORTANT: ensure correct column name exists
    # expected: Fund, Year, Sector, L2_Distance

    df = df.dropna(subset=['L2_Distance'])

    # ---------------------------
    # Quantiles within sector (same as old script)
    # ---------------------------
    df['Quantile'] = pd.qcut(
        df['L2_Distance'],
        q=10,
        labels=range(1, 11)
    )

    # ---------------------------
    # Save output (same format as old script)
    # ---------------------------
    out_file = os.path.join(output_path, f"{sector}_quantiles_byyear.csv")
    df.to_csv(out_file, index=False)

    print(f"Saved: {out_file}")