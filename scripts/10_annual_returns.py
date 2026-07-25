import os
import pandas as pd
import numpy as np

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FUNDS_DIR = os.path.join(BASE_DIR, "../funds")

# Updated sector names to match actual folders
sectors = ["energy", "technology", "healthcare", "utilities", "real estate"]

# -----------------------------
# HELPERS
# -----------------------------
def compute_annual_return(series):
    """Calculate annual return from daily returns"""
    return (1 + series).prod() - 1

# -----------------------------
# MAIN LOOP
# -----------------------------
for sector in sectors:
    print(f"\nProcessing {sector.upper()}...")

    sector_path = os.path.join(FUNDS_DIR, sector)
    
    if not os.path.exists(sector_path):
        print(f"  Path not found: {sector_path}")
        continue

    output_path = os.path.join(sector_path, "annual")
    os.makedirs(output_path, exist_ok=True)

    files = [f for f in os.listdir(sector_path) if f.endswith(".csv")]
    
    # Skip non-fund files
    files = [f for f in files if not any(x in f for x in ['_mask', '_actual_days', 'imputation', 'justification', 'log'])]
    # Skip SPY benchmark
    files = [f for f in files if f != 'SPY_1999_2025.csv']

    all_funds = []

    # -----------------------------
    # EACH FUND
    # -----------------------------
    for file in files:
        fund_name = file.split("_")[0]
        
        # Skip if fund_name looks like a description
        if fund_name in ['imputation', 'justification', 'log', 'SPY']:
            continue

        df = pd.read_csv(os.path.join(sector_path, file))
        
        # Check required columns
        if 'Year' not in df.columns or 'Daily_Return' not in df.columns:
            print(f"  Warning: {file} missing required columns")
            continue

        df["Year"] = df["Year"].astype(int)
        df["Daily_Return"] = df["Daily_Return"].astype(float)

        annual = (
            df.groupby("Year")["Daily_Return"]
            .apply(compute_annual_return)
            .reset_index()
        )

        annual["Fund"] = fund_name
        all_funds.append(annual)

    if not all_funds:
        print(f"  No valid fund data found for {sector}")
        continue

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

    print(f"  ✓ {sector} done → saved in annual/")
    print(f"    Funds: {len(pivot.columns)}")
    print(f"    Years: {len(pivot)}")

print("\n" + "="*60)
print("COMPLETE! Annual returns calculated for all sectors.")
print("="*60)