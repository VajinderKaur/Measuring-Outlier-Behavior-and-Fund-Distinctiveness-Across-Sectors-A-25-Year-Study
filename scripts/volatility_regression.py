"""
Volatility Regression: Test if I(P) predicts lower realized volatility.
Uses daily returns (annualized volatility) for accurate estimates.
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLUSTER_DIR = os.path.join(BASE_DIR, "clusters")
FUNDS_DIR = os.path.join(BASE_DIR, "funds")
OUT_DIR = os.path.join(BASE_DIR, "plots", "volatility_regression")
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Hamming functions (copied from your script)
# -----------------------------
def hamming(s1, s2):
    mask = s1.notna() & s2.notna()
    if mask.sum() == 0:
        return np.nan, 0
    mismatch = (s1[mask] != s2[mask]).sum()
    return mismatch, mask.sum()

def portfolio_index_weighted(trajs, weights=None):
    funds = list(trajs.columns)
    m = len(funds)
    if m < 2:
        return 0
    if weights is None:
        weights = [1/m] * m
    total = 0
    for i, f1 in enumerate(funds):
        for j, f2 in enumerate(funds):
            if i < j:
                d, _ = hamming(trajs[f1], trajs[f2])
                if not np.isnan(d):
                    total += weights[i] * weights[j] * d
    return total

def compute_I_for_portfolio(traj, portfolio_funds):
    sub_traj = traj[portfolio_funds]
    return portfolio_index_weighted(sub_traj)

# -----------------------------
# Load cluster assignments (k=10)
# -----------------------------
sector_files = {
    'Technology': os.path.join(CLUSTER_DIR, "technology_quantiles_global.csv"),
    'Healthcare': os.path.join(CLUSTER_DIR, "healthcare_quantiles_global.csv"),
    'Utilities': os.path.join(CLUSTER_DIR, "utilities_quantiles_global.csv"),
    'Energy': os.path.join(CLUSTER_DIR, "energy_quantiles_global.csv"),
    'Real Estate': os.path.join(CLUSTER_DIR, "real_estate_quantiles_global.csv")
}

# -----------------------------
# Load daily returns (robust)
# -----------------------------
all_returns = {}
for sector, fpath in sector_files.items():
    sector_folder = os.path.join(FUNDS_DIR, sector.lower())
    if not os.path.exists(sector_folder):
        print(f"Warning: {sector_folder} not found.")
        continue
    print(f"Loading {sector} returns from {sector_folder}")
    for csv_file in os.listdir(sector_folder):
        if not csv_file.endswith(".csv"):
            continue
        # Skip non-data files
        if any(x in csv_file for x in ['imputation', 'log', 'justification']):
            continue
        fund = csv_file.split('_')[0]
        file_path = os.path.join(sector_folder, csv_file)
        try:
            df = pd.read_csv(file_path)
            # Ensure required columns exist
            if 'Date' not in df.columns or 'Daily_Return' not in df.columns:
                print(f"  Skipping {csv_file}: missing Date or Daily_Return")
                continue
            df['Date'] = pd.to_datetime(df['Date'])
            df['Return'] = df['Daily_Return']
            df = df.dropna(subset=['Return'])
            if len(df) > 0:
                all_returns[fund] = df[['Date', 'Return']]
        except Exception as e:
            print(f"  Error reading {csv_file}: {e}")
            continue

print(f"\nLoaded returns for {len(all_returns)} funds.")

# -----------------------------
# Prepare trajectories (k=10)
# -----------------------------
trajectories = {}
for sector, fpath in sector_files.items():
    if not os.path.exists(fpath):
        continue
    df = pd.read_csv(fpath)
    # Find cluster column
    cluster_col = None
    for col in ['Decile_Cluster', 'Cluster']:
        if col in df.columns:
            cluster_col = col
            break
    if cluster_col is None:
        continue
    df = df.rename(columns={cluster_col: 'Cluster'})
    df = df[df['Fund'] != 'SPY']
    yearly = df.groupby(['Year', 'Fund'], as_index=False)['Cluster'].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
    )
    traj = yearly.pivot(index='Year', columns='Fund', values='Cluster').sort_index()
    trajectories[sector] = traj

print(f"\nPrepared trajectories for {len(trajectories)} sectors.")

# -----------------------------
# Generate random portfolios
# -----------------------------
np.random.seed(42)
N_PORTFOLIOS = 500
MIN_FUNDS = 3
MAX_FUNDS = 6
reg_data = []

for sector, traj in trajectories.items():
    funds = list(traj.columns)
    if len(funds) < MIN_FUNDS:
        continue
    print(f"\nProcessing {sector}: {len(funds)} funds")
    
    # Pre‑compute annualized volatility for each fund
    fund_vols = {}
    for fund in funds:
        if fund in all_returns:
            rets = all_returns[fund]['Return'].dropna()
            if len(rets) > 0:
                fund_vols[fund] = rets.std() * np.sqrt(252)
            else:
                fund_vols[fund] = np.nan
        else:
            fund_vols[fund] = np.nan
    
    available = [f for f in funds if f in fund_vols and not np.isnan(fund_vols[f])]
    if len(available) < MIN_FUNDS:
        continue
    
    for _ in range(N_PORTFOLIOS):
        n = np.random.randint(MIN_FUNDS, min(MAX_FUNDS, len(available)) + 1)
        chosen = np.random.choice(available, size=n, replace=False)
        chosen = list(chosen)
        
        I_p = compute_I_for_portfolio(traj, chosen)
        if np.isnan(I_p) or I_p == 0:
            continue
        
        # Merge daily returns for chosen funds
        port_ret = None
        for fund in chosen:
            ret_df = all_returns[fund][['Date', 'Return']].rename(columns={'Return': fund})
            if port_ret is None:
                port_ret = ret_df
            else:
                port_ret = pd.merge(port_ret, ret_df, on='Date', how='inner')
        if port_ret is None or len(port_ret) < 60:
            continue
        port_ret['Port_Return'] = port_ret[chosen].mean(axis=1)
        port_vol = port_ret['Port_Return'].std() * np.sqrt(252)
        avg_vol = np.nanmean([fund_vols[f] for f in chosen])
        
        # Ensure all values are finite and valid
        if not np.isfinite(I_p) or not np.isfinite(port_vol) or not np.isfinite(avg_vol):
            continue
        
        reg_data.append({
            'Sector': sector,
            'I_P': I_p,
            'Portfolio_Volatility': port_vol,
            'Avg_Constituent_Vol': avg_vol,
            'N_Funds': n
        })

# -----------------------------
# Run regression (with comprehensive data cleaning)
# -----------------------------
if not reg_data:
    print("\nNo portfolio data generated. Check your data.")
else:
    # Convert to DataFrame
    reg_df = pd.DataFrame(reg_data)
    
    print(f"\nInitial portfolio count: {len(reg_df)}")
    
    # ============================================================
    # STEP 1: FORCE ALL NUMERIC COLUMNS TO FLOAT
    # ============================================================
    numeric_cols = ['I_P', 'Avg_Constituent_Vol', 'N_Funds', 'Portfolio_Volatility']
    for col in numeric_cols:
        reg_df[col] = pd.to_numeric(reg_df[col], errors='coerce')
    
    # ============================================================
    # STEP 2: DROP ROWS WITH NaN OR INF IN ANY RELEVANT COLUMN
    # ============================================================
    reg_df = reg_df.dropna(subset=numeric_cols)
    reg_df = reg_df[np.isfinite(reg_df[numeric_cols]).all(axis=1)]
    
    # ============================================================
    # STEP 3: REMOVE DEGENERATE PORTFOLIOS (I_P == 0)
    # ============================================================
    reg_df = reg_df[reg_df['I_P'] > 0]
    
    # ============================================================
    # SIMPLE REGRESSION (Raw effect without Avg_Constituent_Vol)
    # ============================================================
    X_simple = sm.add_constant(reg_df[['I_P']])
    dummies_simple = pd.get_dummies(reg_df['Sector'], drop_first=True).astype(float)
    X_simple = pd.concat([X_simple, dummies_simple], axis=1).astype(float)
    y_simple = reg_df['Portfolio_Volatility'].astype(float)
    model_simple = sm.OLS(y_simple, X_simple).fit()

    print("\n" + "="*60)
    print("SIMPLE REGRESSION: Volatility ~ I(P) + Sector")
    print("="*60)
    print(model_simple.summary())

    print(f"Portfolio count after cleaning: {len(reg_df)}")
    
    if len(reg_df) < 30:
        print("Too few portfolios for meaningful regression. Exiting.")
        exit()
    
    # ============================================================
    # STEP 4: PREPARE REGRESSION MATRIX (ENSURE FLOAT DTYPE)
    # ============================================================
    # Independent variables
    X = reg_df[['I_P', 'Avg_Constituent_Vol', 'N_Funds']].astype(float)
    X = sm.add_constant(X)  # adds 'const' column
    
    # Sector dummies (convert to float to avoid object dtype)
    dummies = pd.get_dummies(reg_df['Sector'], drop_first=True).astype(float)
    X = pd.concat([X, dummies], axis=1)
    
    # Final safety check: ensure everything is float
    X = X.astype(float)
    
    # Dependent variable
    y = reg_df['Portfolio_Volatility'].astype(float)
    
    # ============================================================
    # STEP 5: FIT MODEL
    # ============================================================
    model = sm.OLS(y, X).fit()
    
    print("\n" + "="*60)
    print("REGRESSION RESULTS")
    print("="*60)
    print(model.summary())
    
    # Save summary
    with open(os.path.join(OUT_DIR, "regression_summary.txt"), 'w') as f:
        f.write(model.summary().as_text())
    
    # Save coefficients for easy table making
    coef_df = model.params.reset_index()
    coef_df.columns = ['Variable', 'Coefficient']
    coef_df['P_value'] = model.pvalues.values
    coef_df.to_csv(os.path.join(OUT_DIR, "regression_coefficients.csv"), index=False)
    
    # ============================================================
    # STEP 6: SCATTER PLOT (I(P) vs Volatility)
    # ============================================================
    plt.figure(figsize=(10,6))
    plt.scatter(reg_df['I_P'], reg_df['Portfolio_Volatility'], alpha=0.5, s=20)
    
    # Univariate regression line for visualization
    slope, intercept = np.polyfit(reg_df['I_P'], reg_df['Portfolio_Volatility'], 1)
    x_vals = np.linspace(reg_df['I_P'].min(), reg_df['I_P'].max(), 100)
    plt.plot(x_vals, intercept + slope*x_vals, 'r-', linewidth=2, 
             label=f'OLS slope = {slope:.4f} (p={model.pvalues["I_P"]:.4f})')
    
    plt.xlabel("Hamming Index I(P)", fontsize=12)
    plt.ylabel("Annualized Portfolio Volatility", fontsize=12)
    plt.title("I(P) vs Realized Volatility (all sectors)", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "I_vs_volatility.jpg"), dpi=300)
    plt.close()
    
    print(f"\n✓ Coefficient table saved to: {os.path.join(OUT_DIR, 'regression_coefficients.csv')}")
    print(f"✓ Scatter plot saved to: {os.path.join(OUT_DIR, 'I_vs_volatility.jpg')}")
    print("✓ Done.")

    