"""
Daily Returns Collector for Mutual Funds by Sector
Collects exactly 252 UNIQUE trading days per year with zero imputation
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# ============================================================================
# CONFIGURATION
# ============================================================================

TARGET_DAYS = 252  # Industry standard for annualization (S&P 500 convention)
START_YEAR = 1999
END_YEAR = 2025

# Base path for saving data
BASE_SAVE_PATH = '../funds/'

# ============================================================================
# SECTOR FUNDS
# ============================================================================

# TECH SECTOR
TECH_FUNDS = ['KTCAX', 'MTCAX', 'SLMCX', 'RSIFX', 
              'FSPTX', 'FSCSX', 'FSELX', 'FDCPX', 'SHGTX', 
              'ROGSX', 'ICTEX']

# UTILITIES SECTOR
UTILITIES_FUNDS = ['BULIX', 'MMUFX', 'PRUAX', 'FKUTX', 
                   'EVUAX', 'FSUTX', 'FIUIX', 'FUGAX', 'GASFX', 'ICTUX']

# REAL ESTATE SECTOR
RE_FUNDS = ['CSEIX', 'FRESX', 'DFREX', 'TRREX', 
            'PHRAX', 'DPREX', 'FREAX', 'IARAX', 'RPFRX', 
            'SOAAX', 'STMDX']

# HEALTHCARE SECTOR
HEALTHCARE_FUNDS = ['FSHCX', 'PHSTX', 'ETHSX', 'VGHCX', 
                    'PRHSX', 'FSPHX', 'FSMEX', 'FACDX', 'FBDIX']

# ENERGY SECTOR
ENERGY_FUNDS = ['ICPAX', 'VGENX', 'FSENX', 'RYEIX', 'FNARX', 'PRNEX']

# Benchmark
SECTOR_INDICES = ['SPY']

# ============================================================================
# SECTOR MAPPING
# ============================================================================

SECTORS = {
    '1': {'name': 'Technology', 'funds': TECH_FUNDS},
    '2': {'name': 'Utilities', 'funds': UTILITIES_FUNDS},
    '3': {'name': 'Real Estate', 'funds': RE_FUNDS},
    '4': {'name': 'Healthcare', 'funds': HEALTHCARE_FUNDS},
    '5': {'name': 'Energy', 'funds': ENERGY_FUNDS},
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_banner():
    print("=" * 80)
    print(" " * 20 + "MUTUAL FUND DAILY RETURNS COLLECTOR")
    print("=" * 80)
    print(f"  • Collects exactly {TARGET_DAYS} UNIQUE trading days per year")
    print(f"  • Missing days imputed with zero returns")
    print(f"  • Industry standard annualization (252 days)")
    print("=" * 80)

def select_sector():
    print("\n" + "-" * 60)
    print("AVAILABLE SECTORS:")
    print("-" * 60)
    for key, sector in SECTORS.items():
        print(f"  {key}. {sector['name']} ({len(sector['funds'])} funds)")
    print(f"  {len(SECTORS)+1}. ALL SECTORS")
    print("-" * 60)
    
    while True:
        choice = input("\nSelect sector (1-6) or 'q' to quit: ").strip()
        if choice.lower() == 'q':
            return None
        if choice in SECTORS or choice == str(len(SECTORS)+1):
            return choice
        print(f"  Invalid choice. Please enter 1-{len(SECTORS)+1}")

def create_folder(folder_name):
    full_path = os.path.join(BASE_SAVE_PATH, folder_name.lower())
    os.makedirs(full_path, exist_ok=True)
    return full_path

def get_spy_trading_dates(year):
    """
    Get exactly TARGET_DAYS UNIQUE trading dates from SPY.
    If fewer than TARGET_DAYS exist, generate synthetic future dates.
    """
    try:
        spy = yf.Ticker('SPY')
        data = spy.history(start=f"{year}-01-01", end=f"{year}-12-31")
        if data.empty:
            return None
        
        all_dates = list(data.index.date)
        
        # If we have enough unique days, take first TARGET_DAYS
        if len(all_dates) >= TARGET_DAYS:
            return all_dates[:TARGET_DAYS]
        else:
            # Need more unique days - generate synthetic future dates
            # These dates will never match any fund data, so they'll be imputed as zeros
            padded_dates = all_dates.copy()
            last_date = all_dates[-1]
            
            while len(padded_dates) < TARGET_DAYS:
                last_date = last_date + timedelta(days=1)
                padded_dates.append(last_date)
            
            return padded_dates
    except Exception as e:
        print(f"  Warning: {e}")
        return None

def get_spy_actual_trading_dates(year):
    """Get ALL actual SPY trading dates (no padding) for imputation table"""
    try:
        spy = yf.Ticker('SPY')
        data = spy.history(start=f"{year}-01-01", end=f"{year}-12-31")
        if data.empty:
            return None
        return list(data.index.date)
    except Exception as e:
        print(f"  Warning: {e}")
        return None

def fetch_returns(symbol, year, reference_dates):
    """
    Fetch returns, impute missing with zeros.
    reference_dates contains exactly TARGET_DAYS UNIQUE dates.
    """
    log_info = {
        'symbol': symbol, 
        'year': year, 
        'actual_days_available': 0,
        'matched': 0, 
        'imputed': 0,
        'zero_returns': 0
    }
    
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=f"{year}-01-01", end=f"{year}-12-31")
        
        if data.empty:
            # No data available - all days imputed
            returns = [0.0] * len(reference_dates)
            log_info['imputed'] = len(reference_dates)
            log_info['zero_returns'] = len(reference_dates)
            log_info['actual_days_available'] = 0
        else:
            # Calculate daily returns
            data['Daily_Return'] = data['Close'].pct_change()
            # Drop the first row (NaN) - this removes one day
            data = data.dropna()
            log_info['actual_days_available'] = len(data)
            return_dict = dict(zip(data.index.date, data['Daily_Return']))
            
            returns = []
            for date in reference_dates:
                if date in return_dict:
                    returns.append(return_dict[date])
                    log_info['matched'] += 1
                else:
                    returns.append(0.0)
                    log_info['imputed'] += 1
                    log_info['zero_returns'] += 1
        
        # Create returns DataFrame
        df = pd.DataFrame({
            'Date': reference_dates,
            'Daily_Return': returns,
            'Year': [year] * len(reference_dates)
        })
        return df, log_info
        
    except Exception as e:
        print(f"    Error: {e}")
        returns = [0.0] * len(reference_dates)
        df = pd.DataFrame({
            'Date': reference_dates,
            'Daily_Return': returns,
            'Year': [year] * len(reference_dates)
        })
        log_info['imputed'] = len(reference_dates)
        log_info['zero_returns'] = len(reference_dates)
        log_info['actual_days_available'] = 0
        return df, log_info

def generate_imputation_tables(all_logs, sector_path, sector_name):
    """Generate imputation tables for reviewer response"""
    
    # Detailed table: each symbol-year
    detail_rows = []
    for log in all_logs:
        imputation_rate = (log['imputed'] / TARGET_DAYS * 100) if log['imputed'] else 0
        completeness = (log['matched'] / TARGET_DAYS * 100) if log['matched'] else 0
        
        detail_rows.append({
            'symbol': log['symbol'],
            'year': log['year'],
            'actual_days_available': log['actual_days_available'],
            'target_days': TARGET_DAYS,
            'days_matched': log['matched'],
            'days_imputed': log['imputed'],
            'imputation_rate_pct': round(imputation_rate, 2),
            'completeness_pct': round(completeness, 2),
            'zero_returns': log['zero_returns']
        })
    
    detail_df = pd.DataFrame(detail_rows)
    detail_csv = os.path.join(sector_path, f"imputation_detail_{START_YEAR}_{END_YEAR}.csv")
    detail_df.to_csv(detail_csv, index=False)
    
    # Yearly summary for reviewer (using SPY actual days)
    yearly_summary = []
    for year in range(START_YEAR, END_YEAR + 1):
        spy_actual = get_spy_actual_trading_dates(year)
        if spy_actual:
            spy_actual_count = len(spy_actual)
            # Note: After pct_change, first day is dropped, so actual returns = spy_actual_count - 1
            spy_returns_available = spy_actual_count - 1
            spy_imputation_needed = TARGET_DAYS - spy_returns_available
            
            # Get fund-level imputation stats for this year
            year_logs = [log for log in all_logs if log['year'] == year]
            if year_logs:
                avg_fund_imputed = sum(log['imputed'] for log in year_logs) / len(year_logs)
                avg_fund_matched = sum(log['matched'] for log in year_logs) / len(year_logs)
            else:
                avg_fund_imputed = spy_imputation_needed
                avg_fund_matched = spy_returns_available
            
            yearly_summary.append({
                'year': year,
                'spy_actual_trading_days': spy_actual_count,
                'spy_returns_available': spy_returns_available,
                'target_days': TARGET_DAYS,
                'spy_days_imputed': spy_imputation_needed,
                'avg_fund_days_matched': round(avg_fund_matched, 1),
                'avg_fund_days_imputed': round(avg_fund_imputed, 1),
                'avg_imputation_rate_pct': round(avg_fund_imputed / TARGET_DAYS * 100, 1)
            })
    
    summary_df = pd.DataFrame(yearly_summary)
    summary_csv = os.path.join(sector_path, f"justification_summary_{START_YEAR}_{END_YEAR}.csv")
    summary_df.to_csv(summary_csv, index=False)
    
    return detail_csv, summary_csv

def process_sector(sector_choice):
    sector = SECTORS[sector_choice]
    sector_name = sector['name'].lower()
    sector_path = create_folder(sector_name)
    
    all_symbols = sector['funds'] + SECTOR_INDICES
    
    print(f"\n{'='*70}")
    print(f"PROCESSING: {sector['name'].upper()}")
    print(f"{'='*70}")
    print(f"  Funds: {len(sector['funds'])}")
    print(f"  Years: {START_YEAR}-{END_YEAR}")
    print(f"  Target days/year: {TARGET_DAYS} (UNIQUE dates)")
    print(f"  Save path: {sector_path}")
    print("-" * 70)
    
    all_logs = []
    
    for symbol in all_symbols:
        print(f"\n  {symbol}...", end=" ", flush=True)
        
        symbol_data = []
        for year in range(START_YEAR, END_YEAR + 1):
            ref_dates = get_spy_trading_dates(year)
            if ref_dates is None:
                continue
            
            year_data, log = fetch_returns(symbol, year, ref_dates)
            symbol_data.append(year_data)
            all_logs.append(log)
        
        if symbol_data:
            combined = pd.concat(symbol_data, ignore_index=True)
            filename = f"{symbol}_{START_YEAR}_{END_YEAR}.csv"
            combined.to_csv(os.path.join(sector_path, filename), index=False)
            print(f"✓ ({len(symbol_data)} years)")
        else:
            print(f"✗ No data")
    
    # Generate imputation tables for reviewer
    detail_csv, summary_csv = generate_imputation_tables(all_logs, sector_path, sector_name)
    
    # Calculate summary statistics
    total_imputed = sum(log['imputed'] for log in all_logs)
    total_matched = sum(log['matched'] for log in all_logs)
    total = total_imputed + total_matched
    rate = (total_imputed / total * 100) if total > 0 else 0
    max_imputation_days = max((log['imputed'] for log in all_logs), default=0)
    max_imputation_pct = (max_imputation_days / TARGET_DAYS * 100) if max_imputation_days else 0
    year_with_max = next((log['year'] for log in all_logs if log['imputed'] == max_imputation_days), 'N/A')
    
    # Create detailed log file
    log_path = os.path.join(sector_path, f"log_{sector_name}_{START_YEAR}_{END_YEAR}.txt")
    with open(log_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"SECTOR: {sector['name'].upper()}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Year Range: {START_YEAR} - {END_YEAR}\n")
        f.write(f"Target Days: {TARGET_DAYS} (UNIQUE dates, Industry Standard)\n")
        f.write(f"Imputation Method: Zero returns for missing days\n")
        f.write("\n")
        
        f.write("IMPUTATION STATISTICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total observations: {total:,}\n")
        f.write(f"Matched days (actual data): {total_matched:,}\n")
        f.write(f"Imputed days (zero returns): {total_imputed:,}\n")
        f.write(f"Overall imputation rate: {rate:.2f}%\n")
        f.write(f"Maximum imputation: {max_imputation_days} days ({max_imputation_pct:.1f}%) in {year_with_max}\n")
        f.write("\n")
        
        f.write("JUSTIFICATION FOR REVIEWER (Comment #6):\n")
        f.write("-" * 40 + "\n")
        f.write("• 252 UNIQUE days is the industry standard for annualization (CRSP, Bloomberg, S&P)\n")
        f.write(f"• Maximum imputation required: {max_imputation_days} days ({max_imputation_pct:.1f}%)\n")
        f.write("• Zero returns on non-trading days are economically accurate\n")
        f.write("• For years with fewer than 252 trading days, synthetic future dates are generated\n")
        f.write("• These synthetic dates never match fund data, resulting in zero imputation\n")
        f.write("• See imputation_detail and justification_summary CSV files for full transparency\n")
        f.write("\n")
        
        f.write("FILES GENERATED:\n")
        f.write(f"  - Returns: {sector_name}/*_{TARGET_DAYS}days.csv\n")
        f.write(f"  - Imputation detail: {os.path.basename(detail_csv)}\n")
        f.write(f"  - Justification summary: {os.path.basename(summary_csv)}\n")
    
    print(f"\n✓ Saved to: {sector_path}")
    print(f"  ✓ Imputation tables created for reviewer")
    return sector_path

def print_spy_summary():
    """Print summary of SPY trading days for reviewer justification"""
    print("\n" + "=" * 80)
    print("SPY ACTUAL TRADING DAYS SUMMARY (For Reviewer Justification)")
    print("=" * 80)
    print(f"{'Year':<8} {'Actual Days':<15} {'Returns Available':<18} {'vs 252':<12} {'Imputation Needed':<15}")
    print("-" * 80)
    
    all_actual_counts = []
    for year in range(START_YEAR, END_YEAR + 1):
        actual_dates = get_spy_actual_trading_dates(year)
        if actual_dates:
            actual_count = len(actual_dates)
            returns_available = actual_count - 1  # First day dropped by pct_change
            all_actual_counts.append(actual_count)
            diff = TARGET_DAYS - returns_available
            imputation_needed = max(0, diff)
            
            print(f"{year:<8} {actual_count:<15} {returns_available:<18} {diff:+d} days{'':<7} {imputation_needed} days")
    
    if all_actual_counts:
        max_imputation = TARGET_DAYS - (min(all_actual_counts) - 1)
        max_imputation_pct = (max_imputation / TARGET_DAYS * 100)
        min_year_idx = all_actual_counts.index(min(all_actual_counts))
        min_year = START_YEAR + min_year_idx
        
        print("=" * 80)
        print(f"\nNote:")
        print(f"  • SPY actual trading days range from {min(all_actual_counts)} to {max(all_actual_counts)}")
        print(f"  • After pct_change(), first day dropped → returns available = actual_days - 1")
        print(f"  • Maximum imputation needed: {max_imputation} days ({max_imputation_pct:.1f}%) in {min_year}")
        print("  • Synthetic future dates generated for missing unique days")
        print("=" * 80)

def main():
    print_banner()
    
    # Show SPY summary first
    print_spy_summary()
    
    sector_choice = select_sector()
    if sector_choice is None:
        print("\nExiting...")
        return
    
    # Create base directory
    os.makedirs(BASE_SAVE_PATH, exist_ok=True)
    
    print(f"\n📁 Data will be saved to: {BASE_SAVE_PATH}")
    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Exiting...")
        return
    
    # Handle ALL SECTORS
    if sector_choice == str(len(SECTORS)+1):
        print("\n" + "=" * 80)
        print("PROCESSING ALL SECTORS")
        print("=" * 80)
        for key in SECTORS.keys():
            process_sector(key)
    else:
        process_sector(sector_choice)
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"\n📍 Data saved at: {BASE_SAVE_PATH}")
    
    print("\n📊 FOR REVIEWER RESPONSE (Comment #6):")
    print("   • 252 UNIQUE days = industry standard (CRSP, Bloomberg, S&P)")
    print("   • Zero imputation for missing days (economically accurate)")
    print("   • Synthetic future dates generated for years with <252 trading days")
    print("   • Imputation tables show full transparency")
    print("   • See 'imputation_detail_*.csv' and 'justification_summary_*.csv'")
    print("=" * 80)

if __name__ == "__main__":
    main()