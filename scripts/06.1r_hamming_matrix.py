import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# -----------------------------
# USER INPUT: Choose sector(s)
# -----------------------------
print("="*60)
print("ROBUSTNESS HAMMING DISTANCE ANALYSIS")
print("="*60)

print("\nAvailable sectors:")
print("  1. All sectors")
print("  2. Technology")
print("  3. Healthcare")
print("  4. Utilities")
print("  5. Energy")
print("  6. Real Estate")
print("  7. Select specific sectors (comma-separated)")

sector_choice = input("\nEnter your choice (1-7): ").strip()

# Define sectors
ALL_SECTORS = ['Technology', 'Healthcare', 'Utilities', 'Energy', 'Real Estate']
SECTOR_MAP = {
    'technology': 'Technology',
    'healthcare': 'Healthcare',
    'utilities': 'Utilities',
    'energy': 'Energy',
    'real estate': 'Real Estate'
}

if sector_choice == '1':
    SELECTED_SECTORS = ALL_SECTORS
elif sector_choice in ['2', '3', '4', '5', '6']:
    idx = int(sector_choice) - 2
    SELECTED_SECTORS = [ALL_SECTORS[idx]]
elif sector_choice == '7':
    custom_sectors = input("Enter sector names (comma-separated, e.g., Technology, Healthcare): ").strip()
    SELECTED_SECTORS = []
    for s in custom_sectors.split(','):
        s_clean = s.strip().lower()
        if s_clean in SECTOR_MAP:
            SELECTED_SECTORS.append(SECTOR_MAP[s_clean])
        else:
            print(f"  Warning: '{s.strip()}' not recognized. Skipping.")
else:
    print("Invalid choice. Defaulting to all sectors.")
    SELECTED_SECTORS = ALL_SECTORS

print(f"\n✓ Selected sectors: {SELECTED_SECTORS}")

# Ask which k values to process
print("\n" + "="*60)
print("K VALUE SELECTION")
print("="*60)
print("\nAvailable robustness k values: [5, 8, 9, 11, 12]")
print("Or enter custom k values (comma-separated, e.g., 4,6,7,13,15):")

custom_input = input("\nPress Enter for default [5,8,9,11,12], or type custom values: ").strip()

if custom_input:
    K_VALUES = [int(k.strip()) for k in custom_input.split(',')]
else:
    K_VALUES = [5, 8, 9, 11, 12]

print(f"\n✓ Processing k values: {K_VALUES}")

# -----------------------------
# OUTPUT STRUCTURE
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
robustness_path = os.path.join(BASE_DIR, "robustness_clusters")
base_out = os.path.join(BASE_DIR, "plots", "hammingdistance_robustness")
os.makedirs(base_out, exist_ok=True)

# -----------------------------
# HAMMING FUNCTIONS
# -----------------------------
def hamming(s1, s2):
    """Calculate Hamming distance between two trajectories.
    Returns: number of years where clusters differ."""
    mask = s1.notna() & s2.notna()
    if mask.sum() == 0:
        return np.nan, 0
    
    mismatch_count = (s1[mask] != s2[mask]).sum()
    return mismatch_count, mask.sum()

def weighted_hamming(s1, s2):
    """Weighted Hamming distance using |cluster_i - cluster_j|.
    This accounts for ordinal distance between clusters."""
    mask = s1.notna() & s2.notna()
    if mask.sum() == 0:
        return np.nan, 0
    
    weighted_diff = abs(s1[mask] - s2[mask]).sum()
    return weighted_diff, mask.sum()

def portfolio_index_weighted(trajs, weights=None):
    """
    Calculate weighted portfolio Hamming index as defined in paper Eq (7):
    I(P) = sum_{i<j} w_i * w_j * h(F_i, F_j)
    
    For equal weights (w_i = 1/m), this equals sum(d) / m^2
    """
    funds = list(trajs.columns)
    m = len(funds)
    
    if m < 2:
        return 0
    
    if weights is None:
        weights = [1/m] * m  # Equal weights
    
    total = 0
    for i, f1 in enumerate(funds):
        for j, f2 in enumerate(funds):
            if i < j:
                d, _ = hamming(trajs[f1], trajs[f2])
                if not np.isnan(d):
                    total += weights[i] * weights[j] * d
    
    return total

def portfolio_index_unweighted(trajs):
    """
    Unweighted portfolio Hamming index:
    I'(P) = sum_{i<j} h(F_i, F_j)
    """
    funds = list(trajs.columns)
    m = len(funds)
    
    if m < 2:
        return 0
    
    total = 0
    for f1, f2 in combinations(funds, 2):
        d, _ = hamming(trajs[f1], trajs[f2])
        if not np.isnan(d):
            total += d
    
    return total

def average_pairwise_distance(trajs):
    """
    Average number of years any two funds are in different clusters.
    This is I'(P) / number_of_pairs
    """
    funds = list(trajs.columns)
    m = len(funds)
    
    if m < 2:
        return 0
    
    total = portfolio_index_unweighted(trajs)
    n_pairs = m * (m - 1) / 2
    
    return total / n_pairs

def portfolio_index_weighted_hamming(trajs):
    """Weighted Hamming using |i-j| for robustness check."""
    funds = list(trajs.columns)
    m = len(funds)
    if m < 2:
        return 0
    total = 0
    for f1, f2 in combinations(funds, 2):
        d, _ = weighted_hamming(trajs[f1], trajs[f2])
        if not np.isnan(d):
            total += d
    return total / (m * (m-1) / 2)  # Average weighted distance

# -----------------------------
# PROCESS EACH SECTOR AND K VALUE
# -----------------------------
for sector in SELECTED_SECTORS:
    sector_key = sector.lower()
    
    for k in K_VALUES:
        print("\n" + "="*60)
        print(f"Processing {sector} (k={k})")
        print("="*60)
        
        # Construct file path for this sector and k
        filename = f"{sector_key}_quantiles_global_k{k}.csv"
        filepath = os.path.join(robustness_path, filename)
        
        if not os.path.exists(filepath):
            print(f"  File not found: {filepath}")
            print(f"  Skipping...")
            continue
        
        # Create output directory
        out_dir = os.path.join(base_out, sector, f"k{k}")
        os.makedirs(out_dir, exist_ok=True)
        
        # LOAD DATA
        df = pd.read_csv(filepath)
        
        # The cluster column name depends on k
        cluster_col = f'Cluster_k{k}'
        
        # Rename to standard 'Quantile' for consistency with original code
        df = df.rename(columns={cluster_col: 'Quantile'})
        
        # Remove SPY if present (benchmark shown separately in plots)
        df = df[df['Fund'] != 'SPY']
        
        if df.empty:
            print(f"  No funds (excluding SPY) for {sector}, k={k}")
            continue
        
        # Get modal quantile per year (one value per fund-year)
        yearly = (
            df.groupby(['Year', 'Fund'], as_index=False)['Quantile']
              .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
        )
        
        # Create trajectory matrix (years × funds)
        traj = yearly.pivot(index='Year', columns='Fund', values='Quantile').sort_index()
        
        n_funds = len(traj.columns)
        n_years = len(traj)
        
        print(f"  Funds: {n_funds}")
        print(f"  Years: {n_years}")
        
        # -----------------------------
        # PORTFOLIO INDICES
        # -----------------------------
        
        # Weighted index (as per paper Eq 7, equal weights assumed)
        I_weighted = portfolio_index_weighted(traj)
        
        # Unweighted index (sum of all pairwise distances)
        I_unweighted = portfolio_index_unweighted(traj)
        
        # Average pairwise distance (interpretation: avg years different)
        avg_distance = average_pairwise_distance(traj)
        
        # Weighted Hamming (using |i-j| for robustness)
        I_weighted_hamming = portfolio_index_weighted_hamming(traj)
        
        print(f"\n  Portfolio Indices (k={k}):")
        print(f"    Weighted Index I(P) [Eq 7]: {I_weighted:.4f}")
        print(f"    Unweighted Index I'(P): {I_unweighted:.0f}")
        print(f"    Average pairwise distance (years different): {avg_distance:.2f}")
        print(f"    Weighted Hamming (|i-j|): {I_weighted_hamming:.4f}")
        
        # -----------------------------
        # IMPACT TABLE (Marginal contribution of each fund)
        # -----------------------------
        impact = []
        
        for fund in traj.columns:
            reduced = traj.drop(columns=fund)
            I_minus = portfolio_index_weighted(reduced)
            
            impact.append({
                "Fund": fund,
                "I_without_fund": I_minus,
                "marginal_impact": I_weighted - I_minus
            })
        
        impact_df = (
            pd.DataFrame(impact)
            .set_index("Fund")
            .sort_values("marginal_impact", ascending=False)
        )
        
        impact_df.to_csv(os.path.join(out_dir, "impact_table.csv"))
        
        best_fund = impact_df.index[0]
        worst_fund = impact_df.index[-1]
        
        print(f"\n  Most diversifying fund (+{impact_df.loc[best_fund, 'marginal_impact']:.4f}): {best_fund}")
        print(f"  Least diversifying fund ({impact_df.loc[worst_fund, 'marginal_impact']:.4f}): {worst_fund}")
        
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
        
        # Also save weighted Hamming matrix
        Mmat_weighted = pd.DataFrame(index=funds, columns=funds, dtype=float)
        for f1, f2 in combinations(funds, 2):
            d, _ = weighted_hamming(traj[f1], traj[f2])
            Mmat_weighted.loc[f1, f2] = d
            Mmat_weighted.loc[f2, f1] = d
        np.fill_diagonal(Mmat_weighted.values, 0)
        Mmat_weighted.to_csv(os.path.join(out_dir, "hamming_matrix_weighted.csv"))
        
        # -----------------------------
        # HEATMAP
        # -----------------------------
        plt.figure(figsize=(14, 12))
        
        data = Mmat.values
        vmax = np.nanmax(data)
        
        im = plt.imshow(data, aspect='auto', cmap='YlOrRd', vmin=0, vmax=vmax)
        
        plt.xticks(range(len(funds)), funds, rotation=45, ha='right', fontsize=9)
        plt.yticks(range(len(funds)), funds, fontsize=9)
        
        cbar = plt.colorbar(im)
        cbar.set_label("Hamming Distance (years in different clusters)", fontsize=10)
        
        plt.title(f"{sector} Sector — Hamming Distance Matrix (k={k} clusters)\n"
                  f"I(P) = {I_weighted:.3f} | Avg years different = {avg_distance:.1f} / {n_years} years", 
                  fontsize=12, fontweight='bold', pad=12)
        
        # Add numbers inside cells
        for i in range(len(funds)):
            for j in range(len(funds)):
                val = data[i, j]
                if np.isnan(val) or i == j:
                    continue
                
                # White text on dark cells, black on light
                color = "white" if val > vmax/2 else "black"
                plt.text(j, i, f"{int(val)}", ha="center", va="center", 
                         color=color, fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "heatmap.jpg"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # -----------------------------
        # SUMMARY
        # -----------------------------
        summary_text = f"""
{'='*60}
SECTOR: {sector} (k={k} clusters)
{'='*60}

Number of funds: {n_funds}
Number of years: {n_years}

PORTFOLIO HAMMING INDICES:
  Weighted Index I(P) [Eq 7, equal weights]: {I_weighted:.6f}
  Unweighted Index I'(P) [sum of distances]: {I_unweighted:.0f}
  Average pairwise distance (years different): {avg_distance:.2f} / {n_years} years
  Weighted Hamming (|i-j|) [robustness]: {I_weighted_hamming:.4f}

MOST DIVERSE FUND (highest marginal impact):
  {best_fund} (contributes +{impact_df.loc[best_fund, 'marginal_impact']:.6f})

LEAST DIVERSE FUND (lowest marginal impact):
  {worst_fund} (contributes {impact_df.loc[worst_fund, 'marginal_impact']:.6f})

IMPACT TABLE (sorted by marginal impact):
{impact_df.to_string()}

FILES SAVED:
  {out_dir}/
    - impact_table.csv
    - hamming_matrix.csv
    - hamming_matrix_weighted.csv
    - heatmap.jpg
"""
        
        with open(os.path.join(out_dir, "summary.txt"), "w") as f:
            f.write(summary_text)
        
        print(summary_text)

print("\n" + "="*60)
print("COMPLETE! All robustness Hamming analyses processed.")
print(f"Output directory: {base_out}")
print("="*60)