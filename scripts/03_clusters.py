import os
import glob
import pandas as pd
import numpy as np

base_path = "residuals"
output_path = "clusters"

# create output folder if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# ---------------------------
# Loop through each sector folder
# ---------------------------
for folder in glob.glob(os.path.join(base_path, "residuals_*")):

    sector = os.path.basename(folder).replace("residuals_", "")
    print(f"Processing sector: {sector}")

    all_dfs = []

    # ---------------------------
    # Load all funds in sector
    # ---------------------------
    for filepath in glob.glob(os.path.join(folder, "*.csv")):
        filename = os.path.basename(filepath)
        fund_name = filename.split('_')[0]

        df = pd.read_csv(filepath)

        df['Fund'] = fund_name
        df['Year'] = pd.to_datetime(df['Date']).dt.year

        all_dfs.append(df)

    merged_df = pd.concat(all_dfs, ignore_index=True)
    merged_df = merged_df.dropna(subset=['Residual'])

    # ---------------------------
    # L2 (RMS) per Fund-Year
    # ---------------------------
    yearly_df = (
        merged_df.groupby(['Fund', 'Year'])['Residual']
        .agg(lambda x: np.sqrt((x**2).mean()))  # RMS L2 distance
        .reset_index(name='L2_Distance')
    )

    # ---------------------------
    # Quantiles within sector
    # ---------------------------
    yearly_df['Quantile'] = pd.qcut(
        yearly_df['L2_Distance'],
        q=10,
        labels=range(1, 11)
    )

    # ---------------------------
    # Save into clusters folder
    # ---------------------------
    out_file = os.path.join(output_path, f"{sector}_quantiles_byyear.csv")
    yearly_df.to_csv(out_file, index=False)

    print(f"Saved: {out_file}")