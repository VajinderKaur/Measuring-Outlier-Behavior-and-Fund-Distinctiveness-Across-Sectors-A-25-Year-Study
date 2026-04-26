import os
import pandas as pd
import numpy as np

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FUNDS_DIR = os.path.join(BASE_DIR, "../funds")

sectors = ["energy", "tech", "healthcare", "utilities", "re"]

# -----------------------------
# HELPERS
# -----------------------------
def compute_annual_return(series):
    return (1 + series).prod() - 1

# -----------------------------
# MAIN LOOP
# -----------------------------
for sector in sectors:
    print(f"\nProcessing {sector.upper()}...")

    sector_path = os.path.join(FUNDS_DIR, sector)

    output_path = os.path.join(sector_path, "annual")
    os.makedirs(output_path, exist_ok=True)

    files = [f for f in os.listdir(sector_path) if f.endswith(".csv")]

    all_funds = []

    # -----------------------------
    # EACH FUND
    # -----------------------------
    for file in files:
        fund_name = file.split("_")[0]

        df = pd.read_csv(os.path.join(sector_path, file))

        df["Year"] = df["Year"].astype(int)
        df["Daily_Return"] = df["Daily_Return"].astype(float)

        annual = (
            df.groupby("Year")["Daily_Return"]
            .apply(compute_annual_return)
            .reset_index()
        )

        annual["Fund"] = fund_name
        all_funds.append(annual)

    # -----------------------------
    # COMBINE FUNDS
    # -----------------------------
    combined = pd.concat(all_funds, ignore_index=True)

    pivot = combined.pivot(index="Year", columns="Fund", values="Daily_Return")
    pivot = pivot.sort_index()

    # -----------------------------
    # SECTOR MEDIAN
    # -----------------------------
    sector_median = pivot.median(axis=1)

    median_df = pd.DataFrame({
        "Year": sector_median.index,
        f"{sector.upper()}_Median_Return": sector_median.values
    })

    # -----------------------------
    # SAVE OUTPUTS
    # -----------------------------
    pivot.to_csv(os.path.join(output_path, "fund_annual_returns.csv"))
    median_df.to_csv(os.path.join(output_path, "sector_median_annual_returns.csv"), index=False)

    print(f"{sector} done → saved in annual/")