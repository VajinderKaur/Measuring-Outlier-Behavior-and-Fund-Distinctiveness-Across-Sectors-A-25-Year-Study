import pandas as pd
import yfinance as yf
import os
from datetime import datetime, timedelta

# ============================================================================
# CONFIGURATION
# ============================================================================

TARGET_DAYS = 252
START_YEAR = 1999
END_YEAR = 2025

# Paths
OLD_FILE_PATH = "../funds250days/utilities/FUGAX_1999_2025.csv"  # CHANGE THIS
NEW_FILE_PATH = "../funds/utilities/FUGAX_1999_2025.csv"

# ============================================================================
# FUNCTIONS
# ============================================================================

def get_spy_trading_dates(year):
    """Get exactly TARGET_DAYS UNIQUE trading dates from SPY"""
    try:
        spy = yf.Ticker('SPY')
        data = spy.history(start=f"{year}-01-01", end=f"{year}-12-31")
        if data.empty:
            return None
        all_dates = list(data.index.date)
        
        if len(all_dates) >= TARGET_DAYS:
            return all_dates[:TARGET_DAYS]
        else:
            padded_dates = all_dates.copy()
            last_date = all_dates[-1]
            while len(padded_dates) < TARGET_DAYS:
                last_date = last_date + timedelta(days=1)
                padded_dates.append(last_date)
            return padded_dates
    except Exception as e:
        print(f"  Warning: {e}")
        return None

def fetch_new_data(symbol, year, reference_dates):
    """Try to download fresh data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=f"{year}-01-01", end=f"{year}-12-31")
        
        if data.empty:
            return None, None
        
        data['Daily_Return'] = data['Close'].pct_change()
        data = data.dropna()
        return_dict = dict(zip(data.index.date, data['Daily_Return']))
        
        returns = []
        for date in reference_dates:
            if date in return_dict:
                returns.append(return_dict[date])
            else:
                returns.append(None)  # Mark as missing, will fill from old file
        
        return returns, len(data)
    except Exception as e:
        return None, None

# ============================================================================
# MAIN PROCESS
# ============================================================================

print("=" * 60)
print("MERGING OLD AND NEW FUGAX DATA")
print("=" * 60)

# Load old file if exists
old_df = None
if os.path.exists(OLD_FILE_PATH):
    old_df = pd.read_csv(OLD_FILE_PATH)
    old_df['Date'] = pd.to_datetime(old_df['Date']).dt.date
    old_df['Year'] = pd.to_datetime(old_df['Date']).dt.year
    print(f"✓ Loaded old file: {len(old_df)} rows")
    print(f"  Years in old file: {sorted(old_df['Year'].unique())}")
else:
    print(f"⚠ Old file not found at: {OLD_FILE_PATH}")

print("\n" + "-" * 60)
print("COLLECTING NEW DATA FROM YAHOO")
print("-" * 60)

all_new_data = []
years_with_data = []
years_missing = []

for year in range(START_YEAR, END_YEAR + 1):
    print(f"\n{year}...", end=" ")
    
    ref_dates = get_spy_trading_dates(year)
    if ref_dates is None:
        print("SKIP (no SPY calendar)")
        continue
    
    # Try to download fresh data
    returns, num_days = fetch_new_data('FUGAX', year, ref_dates)
    
    if returns is not None and num_days > 100:  # Has substantial data
        # Use new data
        year_df = pd.DataFrame({
            'Date': ref_dates,
            'Daily_Return': returns,
            'Year': [year] * len(ref_dates)
        })
        all_new_data.append(year_df)
        years_with_data.append(year)
        print(f"✓ NEW data ({num_days} days)")
    else:
        # No new data - will use old file
        years_missing.append(year)
        print(f"⚠ NO new data - will use old file")

print("\n" + "-" * 60)
print("MERGING WITH OLD DATA")
print("-" * 60)

# Now merge with old data for missing years
if old_df is not None:
    final_data = []
    
    for year in range(START_YEAR, END_YEAR + 1):
        ref_dates = get_spy_trading_dates(year)
        if ref_dates is None:
            continue
        
        # Check if we already have new data for this year
        existing_year_data = [d for d in all_new_data if d['Year'].iloc[0] == year]
        
        if existing_year_data:
            # Use the new data we already collected
            final_data.append(existing_year_data[0])
            print(f"{year}: Using NEW data")
        else:
            # Use old data
            old_year_data = old_df[old_df['Year'] == year]
            
            if len(old_year_data) > 0:
                old_return_dict = dict(zip(old_year_data['Date'], old_year_data['Daily_Return']))
                
                returns = []
                for date in ref_dates:
                    if date in old_return_dict:
                        returns.append(old_return_dict[date])
                    else:
                        returns.append(0.0)  # Impute zero
                
                year_df = pd.DataFrame({
                    'Date': ref_dates,
                    'Daily_Return': returns,
                    'Year': [year] * len(ref_dates)
                })
                final_data.append(year_df)
                
                matched = sum(1 for d in ref_dates if d in old_return_dict)
                print(f"{year}: Using OLD data (matched {matched}/252 days)")
            else:
                # No data at all - all zeros
                returns = [0.0] * len(ref_dates)
                year_df = pd.DataFrame({
                    'Date': ref_dates,
                    'Daily_Return': returns,
                    'Year': [year] * len(ref_dates)
                })
                final_data.append(year_df)
                print(f"{year}: NO DATA - all zeros")

# Combine and save
if final_data:
    combined_df = pd.concat(final_data, ignore_index=True)
    
    # Create directory if needed
    os.makedirs(os.path.dirname(NEW_FILE_PATH), exist_ok=True)
    combined_df.to_csv(NEW_FILE_PATH, index=False)
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"✓ Saved merged FUGAX data to: {NEW_FILE_PATH}")
    print(f"  Total rows: {len(combined_df)}")
    
    # Verification
    days_per_year = combined_df.groupby('Year').size()
    print(f"\nVerification (days per year):")
    for year, days in days_per_year.items():
        status = "✓" if days == TARGET_DAYS else "✗"
        print(f"  {year}: {days} days {status}")
    
    if (days_per_year == TARGET_DAYS).all():
        print(f"\n✓ All years have exactly {TARGET_DAYS} days!")
    else:
        print(f"\n⚠ Some years have incorrect number of days")
else:
    print("ERROR: No data collected!")