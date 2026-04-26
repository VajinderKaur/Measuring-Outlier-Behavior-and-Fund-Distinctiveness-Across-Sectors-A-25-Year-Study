import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sector_files = {
    'Technology': "../clusters/tech_quantiles_byyear.csv",
    'Healthcare': "../clusters/healthcare_quantiles_byyear.csv",
    'Utilities': "../clusters/utilities_quantiles_byyear.csv",
    'Energy': "../clusters/energy_quantiles_byyear.csv",
    'Real Estate': "../clusters/re_quantiles_byyear.csv"
}

out_base = "../plots/stackedplots"
os.makedirs(out_base, exist_ok=True)

# map deciles → 5 groups
def map_group(q):
    if q <= 2:
        return "D1-2"
    elif q <= 4:
        return "D3-4"
    elif q <= 6:
        return "D5-6"
    elif q <= 8:
        return "D7-8"
    else:
        return "D9-10"


for sector, file in sector_files.items():

    df = pd.read_csv(file)
    df["Quantile"] = pd.to_numeric(df["Quantile"], errors="coerce")

    df["Group"] = df["Quantile"].apply(map_group)

    # yearly percentage table
    yearly = (
        df.groupby(["Year", "Group"])
          .size()
          .unstack(fill_value=0)
    )

    yearly = yearly.div(yearly.sum(axis=1), axis=0) * 100
    yearly = yearly.sort_index()

    # plot
    plt.figure(figsize=(14, 6))

    bottom = np.zeros(len(yearly))

    colors = {
        "D1-2": None,
        "D3-4": None,
        "D5-6": None,
        "D7-8": None,
        "D9-10": None
    }

    for group in yearly.columns:
        plt.bar(yearly.index, yearly[group], bottom=bottom, label=group)
        bottom += yearly[group].values

    plt.title(f"{sector} - Decile Composition Over Time")
    plt.ylabel("Percentage of Funds")
    plt.xlabel("Year")
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(out_base, f"{sector}_stacked.png"), dpi=300)
    plt.close()