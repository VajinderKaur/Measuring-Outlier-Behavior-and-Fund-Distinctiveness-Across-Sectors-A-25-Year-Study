import os
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Load fund quantiles
# -------------------------
df = pd.read_csv("../clusters/energy_quantiles_byyear.csv")

selected_funds = ["FNARX", "FRNRX"]
df = df[df["Fund"].isin(selected_funds)]

df = df.sort_values(["Fund", "Year"])

# -------------------------
# Load SPY (market benchmark)
# -------------------------
spy = pd.read_csv(
    "/projectnb/aisearch/Finance/benchmarks/SPY_1999_2025.csv",
    parse_dates=["Date"]
)

spy["Year"] = spy["Date"].dt.year

# -------------------------
# Yearly L2 (RMS volatility proxy)
# -------------------------
spy_yearly = (
    spy.groupby("Year")["Daily_Return"]
    .agg(lambda x: (x**2).mean() ** 0.5)
    .reset_index(name="L2_Distance")
)

# -------------------------
# FORCE: SPY is always Cluster 1 (market baseline)
# -------------------------
spy_yearly["Quantile"] = 1
spy_yearly["Fund"] = "S&P 500 Index"

# -------------------------
# Combine fund + SPY
# -------------------------
df_all = pd.concat(
    [df[["Fund", "Year", "Quantile"]],
     spy_yearly[["Fund", "Year", "Quantile"]]],
    ignore_index=True
)

df_all = df_all.sort_values(["Fund", "Year"])
years = sorted(df_all["Year"].unique())

# -------------------------
# Colors
# -------------------------
color_map = {
    "FNARX": "#0000FF",
    "FRNRX": "#FF0000",
    "S&P 500 Index": "#000000"
}

# -------------------------
# Plot
# -------------------------
plt.figure(figsize=(12, 6))

# Market baseline line (Cluster 1 reference)
plt.axhline(y=1, color="black", linestyle="--", linewidth=1, alpha=0.7)

for fund, group in df_all.groupby("Fund"):
    plt.plot(
        group["Year"],
        group["Quantile"],
        marker='o',
        linestyle='-',
        label=fund,
        color=color_map.get(fund, None)
    )

plt.xticks(years, rotation=45)
plt.yticks(range(1, 11))
plt.xlabel("Year")
plt.ylabel("L2 Quantile (Fund-Year)")
plt.legend(title="Fund", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle="--", alpha=0.6)

# Save output
os.makedirs("../plots/example", exist_ok=True)
plt.tight_layout()
plt.savefig("../plots/example/energy_frnrx_fnarx_spy.jpg", dpi=1200, bbox_inches="tight")

plt.show()