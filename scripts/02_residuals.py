import pandas as pd
import os
import glob
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Paths
fund_folder = 'Funds/healthcare'
spy_path = 'Benchmarks/SPY_1999_2025.csv'
output_folder = 'residuals/residuals_healthcare'

os.makedirs(output_folder, exist_ok=True)

# Read SPY data
spy_df = pd.read_csv(spy_path, parse_dates=['Date'])
spy_df = spy_df[['Date', 'Daily_Return']]
spy_df = spy_df.rename(columns={'Daily_Return': 'SPY_Return'})

# For trajectory plotting later
residuals_all = []

# Loop over each fund file
for filepath in glob.glob(os.path.join(fund_folder, '*.csv')):
    fund_name = os.path.basename(filepath).split('_')[0]

    # Read fund data
    fund_df = pd.read_csv(filepath, parse_dates=['Date'])
    fund_df = fund_df[['Date', 'Daily_Return']]
    fund_df = fund_df.rename(columns={'Daily_Return': 'Fund_Return'})

    # Merge with SPY on Date
    df = pd.merge(fund_df, spy_df, on='Date', how='inner')

    # Drop missing returns
    df = df.dropna(subset=['Fund_Return', 'SPY_Return'])

    # Regression: Fund ~ SPY
    X = sm.add_constant(df['SPY_Return'])
    y = df['Fund_Return']
    model = sm.OLS(y, X).fit()

    # Residuals
    df['Residual'] = model.resid

    # Save residuals per fund
    df[['Date', 'Fund_Return', 'SPY_Return', 'Residual']].to_csv(
        os.path.join(output_folder, f'{fund_name}_residuals.csv'), index=False
    )

    # Add for plotting
    df['Fund'] = fund_name
    residuals_all.append(df[['Date', 'Residual', 'Fund']])

# Combine all residuals
resid_df = pd.concat(residuals_all, ignore_index=True)
