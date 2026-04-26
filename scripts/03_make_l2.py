import os
import glob
import pandas as pd
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

residual_path = os.path.join(BASE_DIR, "residuals")
l2_path = os.path.join(BASE_DIR, "l2")

os.makedirs(l2_path, exist_ok=True)

for folder in glob.glob(os.path.join(residual_path, "residuals_*")):

    sector = os.path.basename(folder).replace("residuals_", "")
    print(f"Processing L2: {sector}")

    dfs = []

    for file in glob.glob(os.path.join(folder, "*.csv")):

        fund = os.path.basename(file).split('_')[0]
        df = pd.read_csv(file)

        df['Fund'] = fund
        df['Year'] = pd.to_datetime(df['Date']).dt.year

        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.dropna(subset=['Residual'])

    # ---------------------------
    # FUND-YEAR L2
    # ---------------------------
    l2_df = (
        merged.groupby(['Fund', 'Year'])['Residual']
        .agg(lambda x: np.sqrt((x**2).mean()))
        .reset_index(name='L2_Distance')
    )

    l2_df['Sector'] = sector

    out = os.path.join(l2_path, f"{sector}_l2.csv")
    l2_df.to_csv(out, index=False)

    print(f"Saved: {out}")