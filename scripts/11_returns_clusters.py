import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FUNDS_DIR = os.path.join(BASE_DIR, "../funds")
CLUSTER_DIR = os.path.join(BASE_DIR, "../clusters")
OUT_PLOT_DIR = os.path.join(BASE_DIR, "../plots/cluster_vs_returns")
os.makedirs(OUT_PLOT_DIR, exist_ok=True)

# -----------------------------
# CONFIG
# -----------------------------
sector_files = {
    "Energy":     "energy_quantiles_byyear.csv",
    "Tech":       "tech_quantiles_byyear.csv",
    "Healthcare": "healthcare_quantiles_byyear.csv",
    "Utilities":  "utilities_quantiles_byyear.csv",
    "RE":         "re_quantiles_byyear.csv",
}

sector_annual_dirs = {
    "Energy":     "energy",
    "Tech":       "tech",
    "Healthcare": "healthcare",
    "Utilities":  "utilities",
    "RE":         "re",
}

SECTOR_COLORS = {
    "Energy":     "#1f77b4",
    "Tech":       "#ff7f0e",
    "Healthcare": "#2ca02c",
    "Utilities":  "#d62728",
    "RE":         "#9467bd",
}

# -----------------------------
# LOAD & MERGE ALL SECTORS
# -----------------------------
all_records = []

for sector, cluster_file in sector_files.items():

    # --- cluster assignments ---
    cluster_path = os.path.join(CLUSTER_DIR, cluster_file)
    cluster_df = pd.read_csv(cluster_path)
    cluster_df["Year"]     = cluster_df["Year"].astype(int)
    cluster_df["Quantile"] = cluster_df["Quantile"].astype(int)

    # mode per (Year, Fund) in case of duplicates
    cluster_mode = (
        cluster_df.groupby(["Year", "Fund"])["Quantile"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
        .rename(columns={"Quantile": "Cluster"})
    )

    # --- annual returns ---
    annual_path = os.path.join(
        FUNDS_DIR, sector_annual_dirs[sector], "annual", "fund_annual_returns.csv"
    )
    returns_wide = pd.read_csv(annual_path, index_col=0)
    returns_wide.index.name = "Year"
    returns_wide.index = returns_wide.index.astype(int)

    # melt wide → long
    returns_long = (
        returns_wide.reset_index()
        .melt(id_vars="Year", var_name="Fund", value_name="Annual_Return")
        .dropna(subset=["Annual_Return"])
    )

    # --- join ---
    merged = returns_long.merge(cluster_mode, on=["Year", "Fund"], how="inner")
    merged["Sector"] = sector
    all_records.append(merged)

combined = pd.concat(all_records, ignore_index=True)

# -----------------------------
# IDENTIFY & REMOVE OUTLIERS
# -----------------------------
outlier_mask = (combined["Annual_Return"] * 100).abs() > 200
outliers = combined[outlier_mask][["Sector", "Fund", "Year", "Cluster", "Annual_Return"]].copy()
outliers["Annual_Return_%"] = (outliers["Annual_Return"] * 100).round(1)

print("Outliers removed:")
print(outliers[["Sector", "Fund", "Year", "Cluster", "Annual_Return_%"]].to_string(index=False))

plot_df = combined[~outlier_mask].copy()

# shared cluster ids after filtering
cluster_ids = sorted(plot_df["Cluster"].unique())

# -----------------------------
# PLOT 1 — ALL SECTORS COMBINED
# -----------------------------
fig, ax = plt.subplots(figsize=(11, 7))

for sector, grp in plot_df.groupby("Sector"):
    ax.scatter(
        grp["Annual_Return"] * 100,
        grp["Cluster"],
        color=SECTOR_COLORS[sector],
        label=sector,
        alpha=0.65,
        edgecolors="white",
        linewidths=0.4,
        s=55,
    )

ax.set_yticks(cluster_ids)
ax.set_yticklabels([f"Cluster {c}" for c in cluster_ids], fontsize=9)
ax.axvline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
ax.set_xlabel("Annual Return (%)", fontsize=11)
ax.set_ylabel("Cluster ID", fontsize=11)
ax.set_title("Annual Returns vs Cluster Assignment — All Sectors", fontsize=13, fontweight="bold")
ax.legend(title="Sector", framealpha=0.9, fontsize=9)

# -----------------------------
# FOOTNOTE
# -----------------------------
footnote_lines = []
for _, row in outliers.iterrows():
    footnote_lines.append(
        f"{row['Sector']} · {row['Fund']} ({int(row['Year'])}) — "
        f"Cluster {row['Cluster']}, Return: {row['Annual_Return_%']}%"
    )

footnote_text = (
    "* Outliers excluded (|return| > 200%):\n  "
    + "\n  ".join(footnote_lines)
)

fig.text(
    0.01, -0.04,
    footnote_text,
    fontsize=7.5,
    color="dimgray",
    verticalalignment="top",
)

plt.tight_layout()
plt.savefig(
    os.path.join(OUT_PLOT_DIR, "all_sectors_returns_vs_cluster.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()
print("Saved: all_sectors_returns_vs_cluster.png")

# -----------------------------
# PLOT 2 — FACETED (one panel per sector)
# -----------------------------
sectors = list(sector_files.keys())
n = len(sectors)
ncols = 3
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=True)
axes = axes.flatten()

for i, sector in enumerate(sectors):
    ax = axes[i]
    grp = plot_df[plot_df["Sector"] == sector]

    ax.scatter(
        grp["Annual_Return"] * 100,
        grp["Cluster"],
        color=SECTOR_COLORS[sector],
        alpha=0.7,
        edgecolors="white",
        linewidths=0.4,
        s=50,
    )

    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_yticks(cluster_ids)
    ax.set_yticklabels([f"Cluster {c}" for c in cluster_ids], fontsize=8)
    ax.set_title(sector, fontsize=11, fontweight="bold", color=SECTOR_COLORS[sector])
    ax.set_xlabel("Annual Return (%)", fontsize=9)
    ax.set_ylabel("Cluster ID", fontsize=9)

# hide unused panels
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle(
    "Annual Returns vs Cluster Assignment by Sector",
    fontsize=14, fontweight="bold", y=1.02
)

# footnote on faceted plot too
fig.text(
    0.01, -0.03,
    footnote_text,
    fontsize=7.5,
    color="dimgray",
    verticalalignment="top",
)

plt.tight_layout()
plt.savefig(
    os.path.join(OUT_PLOT_DIR, "faceted_returns_vs_cluster.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()
print("Saved: faceted_returns_vs_cluster.png")


# -----------------------------
# VOLATILITY VS CLUSTER
# -----------------------------

# compute annual volatility per fund per year using
# a 3-year rolling std of annual returns
volatility_records = []

for sector, cluster_file in sector_files.items():

    annual_path = os.path.join(
        FUNDS_DIR, sector_annual_dirs[sector], "annual", "fund_annual_returns.csv"
    )
    returns_wide = pd.read_csv(annual_path, index_col=0)
    returns_wide.index.name = "Year"
    returns_wide.index = returns_wide.index.astype(int)

    # rolling 3-year std per fund
    vol_wide = returns_wide.rolling(window=3, min_periods=2).std() * 100

    # melt → long
    vol_long = (
        vol_wide.reset_index()
        .melt(id_vars="Year", var_name="Fund", value_name="Volatility")
        .dropna(subset=["Volatility"])
    )
    vol_long["Year"] = vol_long["Year"].astype(int)

    # load cluster mode for this sector (already computed above,
    # but re-derive here so this block is self-contained)
    cluster_path = os.path.join(CLUSTER_DIR, cluster_file)
    cluster_df   = pd.read_csv(cluster_path)
    cluster_df["Year"]     = cluster_df["Year"].astype(int)
    cluster_df["Quantile"] = cluster_df["Quantile"].astype(int)
    cluster_mode = (
        cluster_df.groupby(["Year", "Fund"])["Quantile"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
        .rename(columns={"Quantile": "Cluster"})
    )

    merged = vol_long.merge(cluster_mode, on=["Year", "Fund"], how="inner")
    merged["Sector"] = sector
    volatility_records.append(merged)

vol_combined = pd.concat(volatility_records, ignore_index=True)

# remove extreme volatility outliers (same logic as returns)
vol_outlier_mask = vol_combined["Volatility"] > 150
vol_outliers     = vol_combined[vol_outlier_mask].copy()
vol_plot_df      = vol_combined[~vol_outlier_mask].copy()

vol_cluster_ids  = sorted(vol_plot_df["Cluster"].unique())

# footnote for vol outliers
if len(vol_outliers) > 0:
    vol_footnote_lines = [
        f"{r['Sector']} · {r['Fund']} ({int(r['Year'])}) — "
        f"Cluster {r['Cluster']}, Vol: {r['Volatility']:.1f}%"
        for _, r in vol_outliers.iterrows()
    ]
    vol_footnote = (
        "* Outliers excluded (volatility > 150%):\n  "
        + "\n  ".join(vol_footnote_lines)
    )
else:
    vol_footnote = "* No volatility outliers excluded."

# -----------------------------
# VOLATILITY PLOT 1 — COMBINED
# -----------------------------
fig, ax = plt.subplots(figsize=(11, 7))

for sector, grp in vol_plot_df.groupby("Sector"):
    ax.scatter(
        grp["Volatility"],
        grp["Cluster"],
        color=SECTOR_COLORS[sector],
        label=sector,
        alpha=0.65,
        edgecolors="white",
        linewidths=0.4,
        s=55,
    )

ax.set_yticks(vol_cluster_ids)
ax.set_yticklabels([f"Cluster {c}" for c in vol_cluster_ids], fontsize=9)
ax.axvline(
    vol_plot_df["Volatility"].mean(),
    color="grey", linewidth=0.8, linestyle="--", alpha=0.6,
    label=f"Mean vol ({vol_plot_df['Volatility'].mean():.1f}%)"
)
ax.set_xlabel("3-Year Rolling Volatility (%)", fontsize=11)
ax.set_ylabel("Cluster ID", fontsize=11)
ax.set_title(
    "Volatility vs Cluster Assignment — All Sectors",
    fontsize=13, fontweight="bold"
)
ax.legend(title="Sector", framealpha=0.9, fontsize=9)

fig.text(0.01, -0.04, vol_footnote, fontsize=7.5,
         color="dimgray", verticalalignment="top")

plt.tight_layout()
plt.savefig(
    os.path.join(OUT_PLOT_DIR, "all_sectors_volatility_vs_cluster.png"),
    dpi=300, bbox_inches="tight"
)
plt.close()
print("Saved: all_sectors_volatility_vs_cluster.png")

# -----------------------------
# VOLATILITY PLOT 2 — FACETED
# -----------------------------
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=True)
axes = axes.flatten()

for i, sector in enumerate(sectors):
    ax  = axes[i]
    grp = vol_plot_df[vol_plot_df["Sector"] == sector]

    ax.scatter(
        grp["Volatility"],
        grp["Cluster"],
        color=SECTOR_COLORS[sector],
        alpha=0.7,
        edgecolors="white",
        linewidths=0.4,
        s=50,
    )

    sector_mean_vol = grp["Volatility"].mean()
    ax.axvline(
        sector_mean_vol,
        color="grey", linewidth=0.8, linestyle="--", alpha=0.6,
        label=f"Mean: {sector_mean_vol:.1f}%"
    )
    ax.legend(fontsize=7, framealpha=0.8)
    ax.set_yticks(vol_cluster_ids)
    ax.set_yticklabels([f"Cluster {c}" for c in vol_cluster_ids], fontsize=8)
    ax.set_title(sector, fontsize=11, fontweight="bold",
                 color=SECTOR_COLORS[sector])
    ax.set_xlabel("3-Year Rolling Volatility (%)", fontsize=9)
    ax.set_ylabel("Cluster ID", fontsize=9)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle(
    "Volatility vs Cluster Assignment by Sector",
    fontsize=14, fontweight="bold", y=1.02
)
fig.text(0.01, -0.03, vol_footnote, fontsize=7.5,
         color="dimgray", verticalalignment="top")

plt.tight_layout()
plt.savefig(
    os.path.join(OUT_PLOT_DIR, "faceted_volatility_vs_cluster.png"),
    dpi=300, bbox_inches="tight"
)
plt.close()
print("Saved: faceted_volatility_vs_cluster.png")