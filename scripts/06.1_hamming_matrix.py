import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# -----------------------------
# OUTPUT STRUCTURE
# -----------------------------
base_out = "../plots/hammingdistance/sectors"
os.makedirs(base_out, exist_ok=True)

sector_files = {
    'Technology': "../clusters/tech_quantiles_byyear.csv",
    'Healthcare': "../clusters/healthcare_quantiles_byyear.csv",
    'Utilities': "../clusters/utilities_quantiles_byyear.csv",
    'Energy': "../clusters/energy_quantiles_byyear.csv",
    'Real Estate': "../clusters/re_quantiles_byyear.csv"
}

# -----------------------------
# HAMMING FUNCTION (RAW COUNT)
# -----------------------------
def hamming(s1, s2):
    mask = s1.notna() & s2.notna()
    if mask.sum() == 0:
        return np.nan, 0

    mismatch_count = (s1[mask] != s2[mask]).sum()
    return mismatch_count, mask.sum()

# -----------------------------
# PORTFOLIO INDEX
# -----------------------------
def portfolio_index(trajs):
    funds = list(trajs.columns)
    M = len(funds)
    weight = 1 / (M * M)

    total = 0

    for f1, f2 in combinations(funds, 2):
        d, _ = hamming(trajs[f1], trajs[f2])
        if not np.isnan(d):
            total += d

    return weight * total

# -----------------------------
# MAIN LOOP
# -----------------------------
for sector, file in sector_files.items():

    print(f"\nProcessing {sector}")

    out_dir = os.path.join(base_out, sector)
    os.makedirs(out_dir, exist_ok=True)

    # LOAD DATA
    df = pd.read_csv(file)
    df['Quantile'] = pd.to_numeric(df['Quantile'], errors='coerce')

    yearly = (
        df.groupby(['Year', 'Fund'], as_index=False)['Quantile']
          .agg(lambda x: x.mode().iloc[0])
    )

    traj = yearly.pivot(index='Year', columns='Fund', values='Quantile').sort_index()

    # PORTFOLIO INDEX
    I_all = portfolio_index(traj)

    # IMPACT TABLE
    impact = []

    for fund in traj.columns:
        reduced = traj.drop(columns=fund)
        I_minus = portfolio_index(reduced)

        impact.append({
            "Fund": fund,
            "I_without_fund": I_minus,
            "marginal_impact": I_all - I_minus
        })

    impact_df = (
        pd.DataFrame(impact)
        .set_index("Fund")
        .sort_values("marginal_impact", ascending=False)
    )

    impact_df.to_csv(os.path.join(out_dir, "impact_table.csv"))

    best_fund = impact_df.index[0]

    # -----------------------------
    # HAMMING MATRIX
    # -----------------------------
    funds = list(traj.columns)
    Mmat = pd.DataFrame(index=funds, columns=funds, dtype=float)

    for f1, f2 in combinations(funds, 2):
        d, _ = hamming(traj[f1], traj[f2])
        Mmat.loc[f1, f2] = d
        Mmat.loc[f2, f1] = d

    np.fill_diagonal(Mmat.values, 0)

    Mmat.to_csv(os.path.join(out_dir, "hamming_matrix.csv"))

    # -----------------------------
    # HEATMAP WITH VALUES (UPDATED)
    # -----------------------------
    plt.figure(figsize=(10, 8))

    data = Mmat.values

    # Light sequential colormap — low = white/yellow, high = dark red
    im = plt.imshow(data, aspect='auto', cmap='YlOrRd')

    plt.xticks(range(len(funds)), funds, rotation=45, ha='right', fontsize=9)
    plt.yticks(range(len(funds)), funds, fontsize=9)

    cbar = plt.colorbar(im)
    cbar.set_label("Hamming Distance (raw count)", fontsize=10)

    plt.title(f"{sector} — Hamming Distance Matrix", fontsize=13, fontweight='bold', pad=12)

    # -----------------------------
    # ADD NUMBERS INSIDE CELLS
    # -----------------------------
    vmin = np.nanmin(data[data > 0]) if np.any(data > 0) else 0
    vmax = np.nanmax(data)
    mid  = (vmin + vmax) / 2

    for i in range(len(funds)):
        for j in range(len(funds)):
            val = data[i, j]

            if np.isnan(val):
                continue

            # Black text on light cells, white on dark cells
            color = "white" if val > mid else "black"

            plt.text(
                j, i,
                f"{int(val)}",
                ha="center",
                va="center",
                color=color,
                fontsize=10,
                fontweight='bold'
            )

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "heatmap.jpg"), dpi=300, bbox_inches='tight')
    plt.close()

    # -----------------------------
    # SUMMARY
    # -----------------------------
    summary_text = f"""
Portfolio Index (no normalization):
I(P) = {I_all:.6f}

Impact Table:
{impact_df.to_string()}

Most Diversifying Fund: {best_fund}
"""

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(summary_text)

    print(summary_text)