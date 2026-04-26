import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime

def create_sector_folder(sector_name):
    folder_name = sector_name.lower().replace(' ', '_')
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created folder: {folder_name}")
    return folder_name

def fetch_yearly_returns_250(symbol, year, target_days=250):
    log_info = {
        'symbol': symbol,
        'year': year,
        'original_days': 0,
        'final_days': target_days,
        'days_added': 0,
        'days_trimmed': 0,
        'zero_returns': 0,
        'zero_prices': 0,
        'status': 'success',
        'notes': []
    }
    
    try:
        # Define date range for the year
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
        # Fetch the data
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        
        if data.empty:
            log_info['status'] = 'no_data'
            log_info['notes'].append(f"No data available for {symbol} in {year}")
            # Create empty dataset with zeros
            date_range = pd.date_range(start=start_date, periods=target_days, freq='D')
            returns_df = pd.DataFrame({
                'Date': date_range.date,
                'Adj_Close': [0.0] * target_days,
                'Daily_Return': [0.0] * target_days,
                'Year': [year] * target_days
            })
            log_info['days_added'] = target_days
            log_info['zero_returns'] = target_days
            log_info['zero_prices'] = target_days
            return returns_df, log_info
        
        # Calculate daily returns
        data['Daily_Return'] = data['Close'].pct_change()
        
        # Create a clean DataFrame with relevant columns
        returns_df = pd.DataFrame({
            'Date': data.index.date,  # Extract date only, no time
            'Adj_Close': data['Close'],
            'Daily_Return': data['Daily_Return'],
            'Year': [year] * len(data)
        })
        
        # Remove the first row (NaN return)
        returns_df = returns_df.dropna()
        log_info['original_days'] = len(returns_df)
        
        # Adjust data to exactly target_days
        if len(returns_df) < target_days:
            # Pad with zeros if we have less than target_days
            shortage = target_days - len(returns_df)
            log_info['days_added'] = shortage
            log_info['notes'].append(f"Padded {shortage} days with zeros due to insufficient data")
            
            # Create padding data
            if len(returns_df) > 0:
                last_date = pd.to_datetime(returns_df['Date'].iloc[-1])
                padding_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                            periods=shortage, freq='D')
            else:
                padding_dates = pd.date_range(start=start_date, periods=shortage, freq='D')
            
            padding_df = pd.DataFrame({
                'Date': padding_dates.date,
                'Adj_Close': [0.0] * shortage,
                'Daily_Return': [0.0] * shortage,
                'Year': [year] * shortage
            })
            
            returns_df = pd.concat([returns_df, padding_df], ignore_index=True)
            
        elif len(returns_df) > target_days:
            # Trim to last target_days if we have more
            excess = len(returns_df) - target_days
            log_info['days_trimmed'] = excess
            log_info['notes'].append(f"Trimmed {excess} oldest days to reach {target_days} days")
            returns_df = returns_df.tail(target_days).reset_index(drop=True)
        
        # Count zeros in final dataset
        log_info['zero_returns'] = (returns_df['Daily_Return'] == 0).sum()
        log_info['zero_prices'] = (returns_df['Adj_Close'] == 0).sum()
        
        # Ensure we have exactly target_days
        assert len(returns_df) == target_days, f"Expected {target_days} days, got {len(returns_df)}"
        
        return returns_df, log_info
        
    except Exception as e:
        log_info['status'] = 'error'
        log_info['notes'].append(f"Error: {str(e)}")
        
        # Create empty dataset with zeros as fallback
        date_range = pd.date_range(start=f"{year}-01-01", periods=target_days, freq='D')
        returns_df = pd.DataFrame({
            'Date': date_range.date,
            'Adj_Close': [0.0] * target_days,
            'Daily_Return': [0.0] * target_days,
            'Year': [year] * target_days
        })
        log_info['days_added'] = target_days
        log_info['zero_returns'] = target_days
        log_info['zero_prices'] = target_days
        return returns_df, log_info

def fetch_multiple_funds_yearly(symbols, start_year, end_year, folder_name, target_days=250):
    results = {}
    log_data = []
    years = range(start_year, end_year + 1)
    total_years = len(years)
    
    for symbol_idx, symbol in enumerate(symbols, 1):
        print(f"\n{'='*60}")
        print(f"Processing {symbol} ({symbol_idx}/{len(symbols)}) for years {start_year}-{end_year}")
        print(f"{'='*60}")
        
        symbol_data = []
        symbol_log_summary = {
            'symbol': symbol,
            'total_years': total_years,
            'successful_years': 0,
            'failed_years': 0,
            'no_data_years': 0,
            'total_days_added': 0,
            'total_days_trimmed': 0,
            'total_zero_returns': 0,
            'total_zero_prices': 0,
            'years_with_issues': []
        }
        
        for year in years:
            print(f"  Fetching {year}...", end="")
            
            year_data, log_info = fetch_yearly_returns_250(symbol, year, target_days)
            log_data.append(log_info)
            
            if year_data is not None:
                symbol_data.append(year_data)
                
                # Update symbol summary
                if log_info['status'] == 'success':
                    symbol_log_summary['successful_years'] += 1
                elif log_info['status'] == 'no_data':
                    symbol_log_summary['no_data_years'] += 1
                else:
                    symbol_log_summary['failed_years'] += 1
                
                symbol_log_summary['total_days_added'] += log_info['days_added']
                symbol_log_summary['total_days_trimmed'] += log_info['days_trimmed']
                symbol_log_summary['total_zero_returns'] += log_info['zero_returns']
                symbol_log_summary['total_zero_prices'] += log_info['zero_prices']
                
                if log_info['days_added'] > 0 or log_info['days_trimmed'] > 0 or log_info['status'] != 'success':
                    symbol_log_summary['years_with_issues'].append(year)
                
                print(f" ✓ ({log_info['original_days']} → {target_days} days)")
                
            else:
                symbol_log_summary['failed_years'] += 1
                print(f" ✗ FAILED")
        
        # Create combined file for this symbol across all years
        if symbol_data:
            combined_symbol_df = pd.concat(symbol_data, ignore_index=True)
            combined_filename = f"{symbol}_{start_year}_{end_year}.csv"
            combined_filepath = os.path.join(folder_name, combined_filename)
            combined_symbol_df.to_csv(combined_filepath, index=False)
            
            results[symbol] = combined_symbol_df
            
            print(f"\n✓ Combined file created: {combined_filename}")
            print(f"  Total rows: {len(combined_symbol_df):,} ({total_years} years × {target_days} days)")
            print(f"  Summary: {symbol_log_summary['successful_years']} successful, {symbol_log_summary['no_data_years']} no-data, {symbol_log_summary['failed_years']} failed years")
            if symbol_log_summary['total_zero_returns'] > 0:
                print(f"  Issues: {symbol_log_summary['total_zero_returns']} zero returns, {symbol_log_summary['total_days_added']} days added, {symbol_log_summary['total_days_trimmed']} days trimmed")
    
    return results, log_data

def create_log_file(log_data, folder_name, start_year, end_year):
    """
    Create detailed log file with processing information
    """
    log_filename = f"processing_log_{start_year}_{end_year}.txt"
    log_filepath = os.path.join(folder_name, log_filename)
    
    with open(log_filepath, 'w') as f:
        f.write("STOCK DATA PROCESSING LOG\n")
        f.write("=" * 50 + "\n")
        f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Year Range: {start_year} - {end_year}\n")
        f.write(f"Target Days Per Year: 250\n\n")
        
        # Group by symbol
        symbols = {}
        for entry in log_data:
            symbol = entry['symbol']
            if symbol not in symbols:
                symbols[symbol] = []
            symbols[symbol].append(entry)
        
        # Write summary for each symbol
        for symbol, entries in symbols.items():
            f.write(f"\n{'='*60}\n")
            f.write(f"SYMBOL: {symbol}\n")
            f.write(f"{'='*60}\n")
            
            # Calculate totals
            total_years = len(entries)
            successful = sum(1 for e in entries if e['status'] == 'success')
            no_data = sum(1 for e in entries if e['status'] == 'no_data')
            errors = sum(1 for e in entries if e['status'] == 'error')
            total_added = sum(e['days_added'] for e in entries)
            total_trimmed = sum(e['days_trimmed'] for e in entries)
            total_zero_returns = sum(e['zero_returns'] for e in entries)
            total_zero_prices = sum(e['zero_prices'] for e in entries)
            
            f.write(f"Summary:\n")
            f.write(f"  Total Years Processed: {total_years}\n")
            f.write(f"  Successful: {successful} years\n")
            f.write(f"  No Data Available: {no_data} years\n")
            f.write(f"  Errors: {errors} years\n")
            f.write(f"  Total Days Added (padding): {total_added}\n")
            f.write(f"  Total Days Trimmed: {total_trimmed}\n")
            f.write(f"  Total Zero Returns: {total_zero_returns}\n")
            f.write(f"  Total Zero Prices: {total_zero_prices}\n\n")
            
            # Year-by-year details
            f.write(f"Year-by-Year Details:\n")
            f.write(f"{'Year':<6} {'Status':<10} {'Orig':<5} {'Added':<6} {'Trimmed':<8} {'ZeroRet':<8} {'ZeroPrice':<10} {'Notes'}\n")
            f.write(f"{'-'*80}\n")
            
            for entry in entries:
                notes_str = "; ".join(entry['notes']) if entry['notes'] else "OK"
                f.write(f"{entry['year']:<6} {entry['status']:<10} {entry['original_days']:<5} "
                       f"{entry['days_added']:<6} {entry['days_trimmed']:<8} {entry['zero_returns']:<8} "
                       f"{entry['zero_prices']:<10} {notes_str}\n")
        
        # Overall summary
        f.write(f"\n{'='*60}\n")
        f.write(f"OVERALL SUMMARY\n")
        f.write(f"{'='*60}\n")
        
        all_symbols = set(e['symbol'] for e in log_data)
        f.write(f"Total Symbols Processed: {len(all_symbols)}\n")
        f.write(f"Total Symbol-Year Combinations: {len(log_data)}\n")
        f.write(f"Successful Fetches: {sum(1 for e in log_data if e['status'] == 'success')}\n")
        f.write(f"No Data Available: {sum(1 for e in log_data if e['status'] == 'no_data')}\n")
        f.write(f"Errors: {sum(1 for e in log_data if e['status'] == 'error')}\n")
        f.write(f"Total Days Added Across All: {sum(e['days_added'] for e in log_data)}\n")
        f.write(f"Total Days Trimmed Across All: {sum(e['days_trimmed'] for e in log_data)}\n")
        f.write(f"Total Zero Returns Across All: {sum(e['zero_returns'] for e in log_data)}\n")
        f.write(f"Total Zero Prices Across All: {sum(e['zero_prices'] for e in log_data)}\n")
    
    return log_filepath

# Main execution
if __name__ == "__main__":
    TARGET_DAYS = 250   
    START_YEAR = 1999
    END_YEAR = 2025
    SECTOR_NAME = "tech"  
    # Define the mutual funds/stocks for this sector
    SECTOR_FUNDS = ['KTCAX', 'WSTAX', 'MTCAX','SLMCX','RSIFX', 
    'FSPTX', 'FSCSX', 'FSELX','FDCPX','SHGTX', 'ROGSX', 'ICTEX'] #tech
        #'BULIX','FRUAX','MMUFX','PRUAX','FKUTX','EVUAX', 'FSUTX','FIUIX', 'FUGAX','GASFX', 'ICTUX'] #Utilities
    #'CSRIX', 'CSEIX', 'FRESX', 'DFREX', 'TRREX', 'PHRAX','DPREX', 'FREAX', 'IARAX', 'RPFRX', 'SOAAX', 'STMDX'] #RE
    #"FSHCX","PHSTX","SWHFX","ETHSX","VGHCX","PRHSX","FSPHX","SHSAX","PHLAX", "FSMEX","SHPAX","FACDX","FBDIX"] #healthcare
    #"ICPAX","VGENX","FSENX","RYEIX","IEFCX","ENPIX", "ENPSX","FNARX", "PRNEX"] #energy ,"FRNRX"
    
    # Define sector indices and benchmarks
    SECTOR_INDICES = [ 'SPY'
        #'^SP500-55',     # S&P 500 Utilities Sector Index (direct index)
        #'XLK'
    ]

    sector_folder = create_sector_folder(SECTOR_NAME)
    all_symbols = SECTOR_FUNDS + SECTOR_INDICES
    
    total_years = END_YEAR - START_YEAR + 1
    print(f"Fetching exactly {TARGET_DAYS} days per year for {SECTOR_NAME.upper()} sector")
    print(f"Processing {len(all_symbols)} symbols for {total_years} years ({START_YEAR}-{END_YEAR})")
    print(f"Expected output: {len(all_symbols)} combined files + 1 log file")
    print("=" * 80)
    results, log_data = fetch_multiple_funds_yearly(all_symbols, START_YEAR, END_YEAR, sector_folder, TARGET_DAYS)
    log_filepath = create_log_file(log_data, sector_folder, START_YEAR, END_YEAR)
    print("\n" + "=" * 80)
    print("FINAL SUMMARY:")
    print(f"Successfully processed: {len(results)} out of {len(all_symbols)} symbols")
    print(f"Each symbol file contains: {total_years * TARGET_DAYS:,} rows ({total_years} years × {TARGET_DAYS} days)")
    print(f"Files saved in '{sector_folder}' folder")
    print(f"Detailed log saved to: {os.path.basename(log_filepath)}")
    
    if results:
        print(f"\nSuccessful symbols:")
        for symbol in results.keys():
            print(f" {symbol}_ALL_YEARS_{START_YEAR}_{END_YEAR}_250days.csv")
    
    failed_symbols = [s for s in all_symbols if s not in results]
    if failed_symbols:
        print(f"\nFailed symbols:")
        for symbol in failed_symbols:
            print(f" {symbol}")
    
    print(f"\n Check the log file for detailed processing information!")

