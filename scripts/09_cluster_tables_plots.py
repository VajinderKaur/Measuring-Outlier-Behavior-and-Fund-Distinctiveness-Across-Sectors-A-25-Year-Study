import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLUSTER_DIR = os.path.join(BASE_DIR, "../clusters")
OUT_TABLE_DIR = os.path.join(BASE_DIR, "../trajectory_table")
OUT_PLOT_DIR = os.path.join(BASE_DIR, "../plots/cluster")

os.makedirs(OUT_TABLE_DIR, exist_ok=True)
os.makedirs(OUT_PLOT_DIR, exist_ok=True)

# -----------------------------
# INPUT FILES
# -----------------------------
sector_files = {
    "Energy": "energy_quantiles_byyear.csv",
    "Tech": "tech_quantiles_byyear.csv",
    "Healthcare": "healthcare_quantiles_byyear.csv",
    "Utilities": "utilities_quantiles_byyear.csv",
    "RE": "re_quantiles_byyear.csv"
}

# -----------------------------
# HELPERS
# -----------------------------
def mode_value(series):
    """
    Deterministic mode:
    - highest frequency wins
    - tie → smallest cluster value wins
    """
    arr = np.asarray(series)
    counts = Counter(arr)

    max_freq = max(counts.values())
    candidates = [k for k, v in counts.items() if v == max_freq]

    return min(candidates)


def compute_switches(series):
    arr = np.asarray(series)
    if len(arr) <= 1:
        return 0
    return np.sum(arr[1:] != arr[:-1])


def compute_inertia(series):
    """
    Longest consecutive run in same cluster
    """
    arr = np.asarray(series)

    if len(arr) == 0:
        return 0

    max_streak = 1
    current = 1

    for i in range(1, len(arr)):
        if arr[i] == arr[i - 1]:
            current += 1
        else:
            max_streak = max(max_streak, current)
            current = 1

    return max(max_streak, current)

# -----------------------------
# STORAGE FOR COMBINED PLOT
# -----------------------------
sector_year_modes = {}

# -----------------------------
# MAIN LOOP
# -----------------------------
for sector, file in sector_files.items():
    print(f"Processing {sector}...")

    path = os.path.join(CLUSTER_DIR, file)
    df = pd.read_csv(path)

    df["Year"] = df["Year"].astype(int)
    df["Quantile"] = df["Quantile"].astype(int)

    # -----------------------------
    # PIVOT: Year x Fund
    # -----------------------------
    pivot = df.pivot(index="Year", columns="Fund", values="Quantile")
    pivot = pivot.sort_index()

    # -----------------------------
    # YEAR MODE (for plot)
    # -----------------------------
    year_mode = pivot.apply(lambda x: mode_value(x.dropna()), axis=1)
    pivot["YEAR_MODE"] = year_mode

    sector_year_modes[sector] = year_mode

    # -----------------------------
    # FUND STATISTICS
    # -----------------------------
    fund_stats = {}

    for fund in pivot.columns:
        if fund == "YEAR_MODE":
            continue

        series = pivot[fund].dropna().values

        if len(series) == 0:
            continue

        fund_stats[fund] = {
            "mode": mode_value(series),
            "median": float(np.median(series)),
            "switches": int(compute_switches(series)),
            "inertia": int(compute_inertia(series))
        }

    # -----------------------------
    # SUMMARY ROWS
    # -----------------------------
    summary_rows = pd.DataFrame(
        index=["mode", "median", "switches", "inertia"],
        columns=pivot.columns
    )

    for fund in fund_stats:
        summary_rows.loc["mode", fund] = fund_stats[fund]["mode"]
        summary_rows.loc["median", fund] = fund_stats[fund]["median"]
        summary_rows.loc["switches", fund] = fund_stats[fund]["switches"]
        summary_rows.loc["inertia", fund] = fund_stats[fund]["inertia"]

    final_table = pd.concat([pivot, summary_rows])

    # -----------------------------
    # SAVE TABLE
    # -----------------------------
    out_path = os.path.join(OUT_TABLE_DIR, f"{sector}_trajectory.csv")
    final_table.to_csv(out_path)

    # -----------------------------
    # PLOT: YEAR MODE
    # -----------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(year_mode.index, year_mode.values, marker="o")

    plt.title(f"{sector} - Yearly Cluster Mode")
    plt.xlabel("Year")
    plt.ylabel("Mode Cluster")

    plot_path = os.path.join(OUT_PLOT_DIR, f"{sector}_mode.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

# -----------------------------
# COMBINED PLOT (ALL SECTORS)
# -----------------------------
plt.figure(figsize=(12, 6))

for sector, series in sector_year_modes.items():
    plt.plot(series.index, series.values, marker="o", label=sector)

plt.title("Sector Cluster Regime Evolution (Yearly Mode)")
plt.xlabel("Year")
plt.ylabel("Mode Cluster")
plt.legend()

combined_path = os.path.join(OUT_PLOT_DIR, "all_sectors_year_mode.png")
plt.savefig(combined_path, bbox_inches="tight")
plt.close()

print("Done: trajectory tables + plots generated.")