import pandas as pd
import matplotlib.pyplot as plt

# Load correct L2-based quantile file
df = pd.read_csv("clusters/tech_quantiles_byyear.csv")

# Fix the two funds you care about
selected_funds = ["ICTEX", "ROGSX"]

df = df[df["Fund"].isin(selected_funds)]

# sort for clean lines
df = df.sort_values(["Fund", "Year"])

years = sorted(df["Year"].unique())

# colors (fixed mapping)
color_map = {
    "ICTEX": "#0000FF",
    "ROGSX": "#FF0000"
}

# plot
plt.figure(figsize=(12, 6))

for fund, group in df.groupby("Fund"):
    plt.plot(
        group["Year"],
        group["Quantile"],
        marker='o',
        linestyle='-',
        label=fund,
        color=color_map[fund]
    )

plt.xticks(years, rotation=45)
plt.yticks(range(1, 11))
plt.xlabel("Year")
plt.ylabel("L2 Quantile (Fund-Year)")
plt.legend(title="Fund", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("plots/tech_ictx_rogsx.jpg", dpi=1200, bbox_inches="tight")
plt.show()