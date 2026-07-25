"""
Compile k‑sensitivity table: most distinctive fund per sector for each k.
Reads summary.txt files from robustness Hamming output.
"""

import os
import pandas as pd
import glob

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROBUST_HAMMING_DIR = os.path.join(BASE_DIR, "plots", "hammingdistance_robustness")
OUTPUT_FILE = os.path.join(BASE_DIR, "plots", "k_sensitivity_table.csv")

sectors = ['Technology', 'Healthcare', 'Utilities', 'Energy', 'Real Estate']
k_values = [5, 8, 9, 11, 12]

rows = []

for sector in sectors:
    for k in k_values:
        summary_path = os.path.join(ROBUST_HAMMING_DIR, sector, f"k{k}", "summary.txt")
        if not os.path.exists(summary_path):
            print(f"  Missing: {summary_path}")
            continue
        
        # Parse the summary file
        with open(summary_path, 'r') as f:
            content = f.read()
        
        # Extract the most diverse fund
        # Look for line like "MOST DIVERSE FUND (highest marginal impact):"
        # Then the next line contains the fund name and contribution
        lines = content.split('\n')
        fund = None
        for i, line in enumerate(lines):
            if 'MOST DIVERSE FUND' in line:
                # The next non-empty line should be the fund
                for j in range(i+1, len(lines)):
                    if lines[j].strip():
                        fund = lines[j].strip().split('(')[0].strip()
                        break
                break
        
        if fund is None:
            print(f"  Could not parse {summary_path}")
            continue
        
        rows.append({
            'Sector': sector,
            'k': k,
            'Most_Distinctive_Fund': fund
        })

# Convert to DataFrame and pivot
df = pd.DataFrame(rows)
pivot = df.pivot(index='Sector', columns='k', values='Most_Distinctive_Fund')
pivot.to_csv(OUTPUT_FILE)
print(f"\n✓ k‑sensitivity table saved to: {OUTPUT_FILE}")
print("\nTable (sectors × k):")
print(pivot)