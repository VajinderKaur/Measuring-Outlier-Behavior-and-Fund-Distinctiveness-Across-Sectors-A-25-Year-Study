import os
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Load fund quantiles (global deciles)
# -------------------------
df = pd.read_csv("/projectnb/aisearch/Finance/clusters/technology_quantiles_global.csv")

selected_funds = ["ICTEX", "ROGSX"]
df = df[df["Fund"].isin(selected_funds)]
df = df.sort_values(["Fund", "Year"])

# -------------------------
# SPY is always C1 (benchmark baseline)
# -------------------------
spy_years = sorted(df["Year"].unique())
spy_df = pd.DataFrame({
    "Fund": ["S&P 500 Index"] * len(spy_years),
    "Year": spy_years,
    "Quantile": [1] * len(spy_years)  # Always C1
})

# -------------------------
# Combine fund + SPY
# -------------------------
df_all = pd.concat([
    df[["Fund", "Year", "Decile_Cluster"]].rename(columns={'Decile_Cluster': 'Quantile'}),
    spy_df
], ignore_index=True)

df_all = df_all.sort_values(["Fund", "Year"])
years = sorted(df_all["Year"].unique())

# -------------------------
# Colors
# -------------------------
color_map = {
    "ICTEX": "#1f77b4",      # Blue
    "ROGSX": "#ff7f0e",      # Orange
    "S&P 500 Index": "#000000"  # Black
}

# -------------------------
# Plot
# -------------------------
plt.figure(figsize=(14, 7))


for fund, group in df_all.groupby("Fund"):
    plt.plot(
        group["Year"],
        group["Quantile"],
        marker='o',
        linestyle='-',
        linewidth=2,
        markersize=6,
        label=fund,
        color=color_map.get(fund, "gray")
    )

plt.xticks(years[::2], rotation=45)
plt.yticks(range(1, 11))
plt.xlabel("Year", fontsize=12)
plt.ylabel("Decile Cluster (C1 = closest to benchmark, C10 = farthest)", fontsize=12)
plt.title("Technology Fund Trajectories vs S&P 500 Benchmark", fontsize=14)
plt.legend(title="Fund", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle="--", alpha=0.4)
plt.ylim(0.5, 10.5)

os.makedirs("../plots/example", exist_ok=True)
plt.tight_layout()
plt.savefig("../plots/example/tech_example_spy.jpg", dpi=300, bbox_inches="tight")

plt.show()
print("✓ Plot saved to: ../plots/example/tech_example_spy.jpg")