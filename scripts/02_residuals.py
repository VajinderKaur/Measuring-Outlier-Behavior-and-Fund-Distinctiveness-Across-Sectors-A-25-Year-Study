"""
Simple 02_residuals.py - No masking, uses ALL 252 days including zeros
- Regressions use ALL days (imputed zeros included)
- Residual vectors are 252 days long
- Simple approach as per professor's instruction
- SPY data loaded from benchmarks/ folder
"""

import pandas as pd
import statsmodels.api as sm
import numpy as np
import os
import glob

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TARGET_DAYS = 252

# ============================================================================
# LOAD SPY DATA (from benchmarks/ folder)
# ============================================================================
spy_df = pd.read_csv(os.path.join(BASE_DIR, "benchmarks", "SPY_1999_2025.csv"), parse_dates=['Date'])
spy_df['Year'] = pd.to_datetime(spy_df['Date']).dt.year
spy_df = spy_df[['Date', 'Year', 'Daily_Return']].rename(columns={'Daily_Return': 'SPY_Return'})

print(f"Loaded SPY data: {len(spy_df)} rows")
print(f"SPY years: {sorted(spy_df['Year'].unique())}")
print("="*60)

# ============================================================================
# INITIALIZE COLLECTORS
# ============================================================================
residuals_all = []
coefficients_all = []

# ============================================================================
# PROCESS EACH FUND - LOOP THROUGH SECTOR SUBFOLDERS
# ============================================================================
for sector_folder in glob.glob(os.path.join(BASE_DIR, "funds", "*")):
    # Skip if not a directory or if it's benchmarks folder
    if not os.path.isdir(sector_folder):
        continue
    if 'benchmarks' in sector_folder:
        continue
    
    sector = os.path.basename(sector_folder)
    print(f"\n{'='*60}")
    print(f"Processing sector: {sector.upper()}")
    print(f"{'='*60}")
    
    # Process each CSV file in this sector folder
    for fund_file in glob.glob(os.path.join(sector_folder, "*.csv")):
        filename = os.path.basename(fund_file)
        
        # Skip non-fund files
        if '_mask' in filename or '_actual_days' in filename:
            continue
        if 'imputation' in filename or 'justification' in filename or 'log' in filename:
            continue
        if filename == 'SPY_1999_2025.csv':
            continue
        
        fund_name = filename.split('_')[0]
        
        print(f"\nProcessing {fund_name} ({sector})")
        
        # Load fund returns (252-day fixed with zeros)
        fund_df = pd.read_csv(fund_file, parse_dates=['Date'])
        fund_df['Year'] = pd.to_datetime(fund_df['Date']).dt.year
        
        # Merge with SPY
        df = pd.merge(fund_df, spy_df, on=['Date', 'Year'], how='inner')
        df = df.dropna(subset=['Daily_Return', 'SPY_Return'])
        
        all_residuals = []
        
        for year, year_df in df.groupby('Year'):
            total_days = len(year_df)
            
            # Check if we have enough days (should be 252, but just in case)
            if total_days < 60:
                print(f"  Warning: {fund_name} {year} only has {total_days} days (need 60), skipping")
                continue
            
            # ================================================================
            # Run regression on ALL days (including zeros!)
            # ================================================================
            X = sm.add_constant(year_df['SPY_Return'])
            y = year_df['Daily_Return']
            model = sm.OLS(y, X).fit()
            
            # Store residuals
            year_residuals = year_df[['Date']].copy()
            year_residuals['Residual'] = model.resid
            year_residuals['Fund'] = fund_name
            year_residuals['Sector'] = sector
            year_residuals['Year'] = year
            
            all_residuals.append(year_residuals)
            
            # Add to master residuals list
            temp_df = year_df[['Date']].copy()
            temp_df['Residual'] = model.resid
            temp_df['Fund'] = fund_name
            temp_df['Sector'] = sector
            residuals_all.append(temp_df)
            
            # Store coefficients
            coef_df = pd.DataFrame([{
                'Fund': fund_name,
                'Sector': sector,
                'Year': year,
                'Total_Days': total_days,
                'Alpha': model.params['const'],
                'Beta': model.params['SPY_Return'],
                'Alpha_Pvalue': model.pvalues['const'],
                'Beta_Pvalue': model.pvalues['SPY_Return'],
                'R_Squared': model.rsquared,
                'Adj_R_Squared': model.rsquared_adj,
                'Residual_Std': model.mse_resid ** 0.5,
                'Fstatistic': model.fvalue,
                'F_Pvalue': model.f_pvalue
            }])
            coefficients_all.append(coef_df)
            
            print(f"  {year}: {total_days} days, Alpha={model.params['const']:.4f}, Beta={model.params['SPY_Return']:.3f}, R²={model.rsquared:.3f}")
        
        # Save individual fund residual file
        if all_residuals:
            result_df = pd.concat(all_residuals, ignore_index=True)
            output_folder = os.path.join(BASE_DIR, "residuals", f"residuals_{sector}")
            os.makedirs(output_folder, exist_ok=True)
            result_df.to_csv(os.path.join(output_folder, f"{fund_name}_residuals.csv"), index=False)
            print(f"  Saved {len(result_df)} residual observations for {fund_name}")

# ============================================================================
# SAVE COMBINED OUTPUTS
# ============================================================================

os.makedirs(os.path.join(BASE_DIR, "residuals"), exist_ok=True)

# Combined residuals file
if residuals_all:
    resid_df = pd.concat(residuals_all, ignore_index=True)
    combined_output = os.path.join(BASE_DIR, "residuals", "all_residuals_combined.csv")
    resid_df.to_csv(combined_output, index=False)
    print(f"\n✓ Combined residuals saved to: {combined_output}")
    print(f"  Total observations: {len(resid_df):,}")
else:
    print("No residuals were generated!")

# Regression coefficients file
if coefficients_all:
    coef_df = pd.concat(coefficients_all, ignore_index=True)
    coef_output = os.path.join(BASE_DIR, "residuals", "regression_coefficients.csv")
    coef_df.to_csv(coef_output, index=False)
    print(f"\n✓ Regression coefficients saved to: {coef_output}")
    print(f"  Total fund-year combinations: {len(coef_df)}")
    
    # Summary statistics
    print("\n" + "="*60)
    print("REGRESSION SUMMARY STATISTICS")
    print("="*60)
    print(f"Average R-squared: {coef_df['R_Squared'].mean():.3f}")
    print(f"Average Alpha: {coef_df['Alpha'].mean():.4f}")
    print(f"Average Beta: {coef_df['Beta'].mean():.3f}")
    print(f"Average days per fund-year: {coef_df['Total_Days'].mean():.1f}")
    print(f"Fund-years with significant alpha (p<0.05): {(coef_df['Alpha_Pvalue'] < 0.05).sum()}/{len(coef_df)}")
    print(f"Fund-years with significant beta (p<0.05): {(coef_df['Beta_Pvalue'] < 0.05).sum()}/{len(coef_df)}")
    
    # Sector-specific averages
    print("\nSector Averages:")
    print("-"*40)
    sector_summary = coef_df.groupby('Sector').agg({
        'Alpha': 'mean',
        'Beta': 'mean',
        'R_Squared': 'mean',
        'Residual_Std': 'mean'
    }).round(4)
    print(sector_summary)

print("\n" + "="*60)
print("KEY METHODOLOGICAL NOTE:")
print("="*60)
print("• Regressions use ALL 252 days (including imputed zeros)")
print("• Simple approach matching industry standard")
print(f"• SPY data loaded from: benchmarks/SPY_1999_2025.csv")
print("="*60)