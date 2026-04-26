import pandas as pd
import os
import glob
import statsmodels.api as sm

# Get project root (one level above scripts/)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

fund_root = os.path.join(BASE_DIR, "funds")
spy_path = os.path.join(BASE_DIR, "benchmarks", "SPY_1999_2025.csv")
output_root = os.path.join(BASE_DIR, "residuals")

os.makedirs(output_root, exist_ok=True)

# Read SPY once
spy_df = pd.read_csv(spy_path, parse_dates=['Date'])
spy_df = spy_df[['Date', 'Daily_Return']].rename(columns={'Daily_Return': 'SPY_Return'})

residuals_all = []

# Loop through all sectors
for fund_folder in glob.glob(os.path.join(fund_root, '*')):

    sector_name = os.path.basename(fund_folder)
    output_folder = os.path.join(output_root, f'residuals_{sector_name}')
    os.makedirs(output_folder, exist_ok=True)

    for filepath in glob.glob(os.path.join(fund_folder, '*.csv')):
        fund_name = os.path.basename(filepath).split('_')[0]

        fund_df = pd.read_csv(filepath, parse_dates=['Date'])
        fund_df = fund_df[['Date', 'Daily_Return']].rename(columns={'Daily_Return': 'Fund_Return'})

        df = pd.merge(fund_df, spy_df, on='Date', how='inner')
        df = df.dropna(subset=['Fund_Return', 'SPY_Return'])

        X = sm.add_constant(df['SPY_Return'])
        y = df['Fund_Return']
        model = sm.OLS(y, X).fit()

        df['Residual'] = model.resid

        df[['Date', 'Fund_Return', 'SPY_Return', 'Residual']].to_csv(
            os.path.join(output_folder, f'{fund_name}_residuals.csv'),
            index=False
        )

        df['Fund'] = fund_name
        df['Sector'] = sector_name
        residuals_all.append(df[['Date', 'Residual', 'Fund', 'Sector']])

resid_df = pd.concat(residuals_all, ignore_index=True)