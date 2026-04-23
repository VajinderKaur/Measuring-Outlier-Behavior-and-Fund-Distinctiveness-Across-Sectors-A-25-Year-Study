import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# -----------------------------
# OUTPUT STRUCTURE
# -----------------------------
base_out = "plots/hammingdistance/sectors"
os.makedirs(base_out, exist_ok=True)

sector_files = {
    'Technology': "clusters/tech_quantiles_byyear.csv",
    'Healthcare': "clusters/hc_quantiles_byyear.csv",
    'Utilities': "clusters/util_quantiles_byyear.csv",
    'Energy': "clusters/energy_quantiles_byyear.csv",
    'Real Estate': "clusters/re_quantiles_byyear.csv"
}

# -----------------------------
# HAMMING FUNCTIONS
# -----------------------------
def hamming(s1, s2):
    mask = s1.notna() & s2.notna()
    n = int(mask.sum())
    if n == 0:
        return np.nan, 0
    return (s1[mask] != s2[mask]).mean(), n


def avg_hamming(trajs):
    funds = list(trajs.columns)
    vals, wts = [], []

    for f1, f2 in combinations(funds, 2):
        d, n = hamming(trajs[f1], trajs[f2])
        if not np.isnan(d):
            vals.append(d)
            wts.append(n)

    return np.average(vals, weights=wts) if vals else np.nan


# -----------------------------
# MAIN LOOP
# -----------------------------
for sector, file in sector_files.items():

    print(f"\nProcessing {sector}")

    out_dir = os.path.join(base_out, sector)
    os.makedirs(out_dir, exist_ok=True)

    # -----------------------------
    # LOAD DATA
    # -----------------------------
    df = pd.read_csv(file)
    df['Quantile'] = pd.to_numeric(df['Quantile'], errors='coerce')

    # -----------------------------
    # TRAJECTORIES (Fund-Year)
    # -----------------------------
    yearly = (
        df.groupby(['Year', 'Fund'], as_index=False)['Quantile']
          .agg(lambda x: x.mode().iloc[0])
    )

    traj = yearly.pivot(index='Year', columns='Fund', values='Quantile').sort_index()

    # -----------------------------
    # OVERALL HAMMING
    # -----------------------------
    H_all = avg_hamming(traj)

    # -----------------------------
    # IMPACT TABLE
    # -----------------------------
    impact = []

    for fund in traj.columns:
        reduced = traj.drop(columns=fund)
        H_minus = avg_hamming(reduced)

        impact.append({
            "Fund": fund,
            "H_without_fund": H_minus,
            "drop_in_diversification": H_all - H_minus
        })

    impact_df = (
        pd.DataFrame(impact)
        .set_index("Fund")
        .sort_values("drop_in_diversification", ascending=False)
    )

    # save table
    impact_df.to_csv(os.path.join(out_dir, "impact_table.csv"))

    best_fund = impact_df.index[0]

    # -----------------------------
    # PAIRWISE HAMMING MATRIX
    # -----------------------------
    funds = list(traj.columns)

    M = pd.DataFrame(index=funds, columns=funds, dtype=float)

    for f1, f2 in combinations(funds, 2):
        d, _ = hamming(traj[f1], traj[f2])
        M.loc[f1, f2] = M.loc[f2, f1] = d

    np.fill_diagonal(M.values, 0)

    M.to_csv(os.path.join(out_dir, "hamming_matrix.csv"))

    # -----------------------------
    # HEATMAP
    # -----------------------------
    plt.figure(figsize=(10, 8))
    plt.imshow(M.values, aspect='auto')
    plt.xticks(range(len(funds)), funds, rotation=90)
    plt.yticks(range(len(funds)), funds)
    plt.colorbar(label="Hamming Distance")
    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, "heatmap.jpg"), dpi=1200)
    plt.close()

    # -----------------------------
    # SUMMARY FILE (IMPORTANT)
    # -----------------------------
    summary_text = f"""
Overall Hamming distance (all funds): {H_all:.3f}

Diversification impact (largest drop = most diversification):
{impact_df.to_string()}

Fund contributing the most diversification: {best_fund}
"""

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(summary_text)

    # -----------------------------
    # CONSOLE OUTPUT
    # -----------------------------
    print(summary_text)