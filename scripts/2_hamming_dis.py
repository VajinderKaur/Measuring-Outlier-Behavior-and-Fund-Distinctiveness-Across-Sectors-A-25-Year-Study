"""
Example: Regular vs Weighted Hamming Distance for Two Technology Funds
ICTEX vs ROGSX - showing the difference between counting differences vs |i-j|
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLUSTER_DIR = os.path.join(BASE_DIR, "../clusters")
OUT_PLOT_DIR = os.path.join(BASE_DIR, "../plots/hamming_example")
os.makedirs(OUT_PLOT_DIR, exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
tech_clusters = pd.read_csv(os.path.join(CLUSTER_DIR, "technology_quantiles_global.csv"))
tech_clusters = tech_clusters.rename(columns={'Decile_Cluster': 'Cluster'})

# Select two example funds (from your paper)
fund1 = "ICTEX"
fund2 = "ROGSX"

print(f"\n{'='*70}")
print(f"EXAMPLE: {fund1} vs {fund2}")
print(f"{'='*70}")

# Get trajectories
traj1 = tech_clusters[tech_clusters['Fund'] == fund1][['Year', 'Cluster']].sort_values('Year')
traj2 = tech_clusters[tech_clusters['Fund'] == fund2][['Year', 'Cluster']].sort_values('Year')

# Merge
merged = pd.merge(traj1, traj2, on='Year', suffixes=('_1', '_2'))

# Calculate distances per year
merged['Regular_Contrib'] = (merged['Cluster_1'] != merged['Cluster_2']).astype(int)
merged['Weighted_Contrib'] = np.abs(merged['Cluster_1'] - merged['Cluster_2'])

# Total distances
total_regular = merged['Regular_Contrib'].sum()
total_weighted = merged['Weighted_Contrib'].sum()

print(f"\nTotal Regular Hamming (binary): {total_regular} years different")
print(f"Total Weighted Hamming (|i-j|): {total_weighted}")
print(f"Average |i-j| when different: {total_weighted/total_regular:.2f} clusters apart")

# ============================================================================
# PLOT 1: Side-by-Side Trajectories with Difference Highlighting
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 8))

ax.plot(merged['Year'], merged['Cluster_1'], 'o-', label=fund1, color='#1f77b4', linewidth=2, markersize=8)
ax.plot(merged['Year'], merged['Cluster_2'], 's-', label=fund2, color='#ff7f0e', linewidth=2, markersize=8)

# Highlight years where they differ
diff_years = merged[merged['Cluster_1'] != merged['Cluster_2']]['Year'].values
for year in diff_years:
    ax.axvspan(year - 0.4, year + 0.4, alpha=0.2, color='red')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Cluster (C1 = closest to benchmark, C10 = farthest)', fontsize=12)
ax.set_title(f'{fund1} vs {fund2}: Cluster Trajectories (1999-2025)', fontsize=14)
ax.set_yticks(range(1, 11))
ax.set_ylim(0.5, 10.5)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left')

ax.annotate(f'Years where clusters differ: {total_regular} out of 27\n(highlighted in red)',
            xy=(2005, 9), xytext=(2002, 7),
            fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(OUT_PLOT_DIR, "example_trajectories.jpg"), dpi=300, bbox_inches='tight')
plt.close()
print(f"\n✓ Saved: example_trajectories.jpg")

# ============================================================================
# PLOT 2: Year-by-Year Distance Comparison (Regular vs Weighted)
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 6))

x = merged['Year']
width = 0.35

bars1 = ax.bar(x - width/2, merged['Regular_Contrib'], width, 
               label=f'Regular Hamming (total={total_regular})', 
               color='steelblue', alpha=0.7, edgecolor='black')

bars2 = ax.bar(x + width/2, merged['Weighted_Contrib'], width, 
               label=f'Weighted Hamming |i-j| (total={total_weighted})', 
               color='coral', alpha=0.7, edgecolor='black')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Distance Contribution', fontsize=12)
ax.set_title(f'{fund1} vs {fund2}: Year-by-Year Hamming Distance', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

ax.annotate(f'Regular: counts 1 regardless of how far apart\nWeighted: counts |i-j| (e.g., C4→C10 = 6)',
            xy=(2005, 6), xytext=(2002, 10),
            fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(OUT_PLOT_DIR, "example_yearly_comparison.jpg"), dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: example_yearly_comparison.jpg")

# ============================================================================
# PLOT 3: Cumulative Distance Over Time
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

merged['Cumulative_Regular'] = merged['Regular_Contrib'].cumsum()
merged['Cumulative_Weighted'] = merged['Weighted_Contrib'].cumsum()

ax.plot(merged['Year'], merged['Cumulative_Regular'], 'o-', 
        label=f'Regular Hamming (final={total_regular})', 
        color='steelblue', linewidth=2, markersize=6)
ax.plot(merged['Year'], merged['Cumulative_Weighted'], 's-', 
        label=f'Weighted Hamming |i-j| (final={total_weighted})', 
        color='coral', linewidth=2, markersize=6)

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Cumulative Hamming Distance', fontsize=12)
ax.set_title(f'{fund1} vs {fund2}: Cumulative Distance Over Time', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

ax.annotate(f'Weighted distance is {total_weighted/total_regular:.1f}x larger\nbecause it accounts for HOW FAR apart clusters are',
            xy=(2015, total_weighted/2), xytext=(2010, total_weighted/2 + 20),
            fontsize=10, style='italic',
            arrowprops=dict(arrowstyle='->', color='red'),
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(OUT_PLOT_DIR, "example_cumulative.jpg"), dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: example_cumulative.jpg")

# ============================================================================
# PLOT 4: Table showing specific years with large differences (FIXED)
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('off')

# Prepare data for table - show only years with differences to make it compact
diff_table = merged[merged['Regular_Contrib'] == 1].copy()
diff_table['Weighted_Contrib'] = diff_table['Weighted_Contrib'].astype(int)

# Create table data
table_data = [['Year', f'{fund1} Cluster', f'{fund2} Cluster', 'Weighted\n(|i-j|)', 'Interpretation']]

for _, row in diff_table.iterrows():
    year = int(row['Year'])
    c1 = int(row['Cluster_1'])
    c2 = int(row['Cluster_2'])
    weighted = int(row['Weighted_Contrib'])
    
    # Add interpretation
    if weighted == 1:
        interpretation = 'Adjacent clusters'
    elif weighted <= 3:
        interpretation = 'Moderate difference'
    else:
        interpretation = 'Large difference!'
    
    table_data.append([
        str(year),
        f'C{c1}',
        f'C{c2}',
        str(weighted),
        interpretation
    ])

# Create table
table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.1, 0.12, 0.12, 0.12, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)

# Style the header row (row 0)
for j in range(len(table_data[0])):
    table[(0, j)].set_facecolor('#4472C4')
    table[(0, j)].set_text_props(weight='bold', color='white', fontsize=11)

# Color code rows by weighted difference
for i in range(1, len(table_data)):
    weighted_val = int(table_data[i][3])
    if weighted_val >= 5:
        color = '#FFCCCC'  # Light red for large differences
    elif weighted_val >= 3:
        color = '#FFE6CC'  # Light orange for moderate differences
    else:
        color = '#CCFFCC'  # Light green for small differences
    
    for j in range(len(table_data[0])):
        table[(i, j)].set_facecolor(color)

# Add title
ax.set_title(f'{fund1} vs {fund2}: Years with Different Clusters (1999-2025)\n'
             f'Regular Hamming: {total_regular} years | Weighted Hamming (|i-j|): {total_weighted}',
             fontsize=14, fontweight='bold', pad=30)

plt.tight_layout()
plt.savefig(os.path.join(OUT_PLOT_DIR, "example_table.jpg"), dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: example_table.jpg")

# ============================================================================
# PLOT 5: Visual explanation of Regular vs Weighted
# ============================================================================
fig, ax = plt.subplots(figsize=(11, 5))
ax.axis('off')

explanation_text = f"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                         REGULAR vs WEIGHTED HAMMING DISTANCE                             ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                          ║
║  Regular Hamming (binary):                                                               ║
║    • Counts 1 if clusters are DIFFERENT (regardless of how far apart)                    ║
║    • C5 vs C6 → 1  (same as C1 vs C10 → 1)                                               ║
║                                                                                          ║
║  Weighted Hamming (|i-j|):                                                               ║
║    • Counts the DISTANCE between clusters: |i - j|                                       ║
║    • C5 vs C6 → 1                                                                        ║
║    • C1 vs C10 → 9  (much larger!)                                                       ║
║                                                                                          ║
║  EXAMPLE: {fund1} vs {fund2}                                                             ║
║    • Regular Hamming: {total_regular} years different                                    ║
║    • Weighted Hamming: {total_weighted}                                                  ║
║    • Weighted is {total_weighted/total_regular:.1f}x larger                              ║
║                                                                                          ║
║  Interpretation:                                                                         ║
║    When these two funds differ, they differ by an average of                             ║
║    {total_weighted/total_regular:.1f} cluster levels.                                    ║
║                                                                                          ║
║  This addresses the reviewer's concern:                                                  ║
║    A difference between C5 and C6 should NOT be treated the same                         ║
║    as a difference between C1 and C10. Weighted Hamming solves this.                     ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
"""

ax.text(0.5, 0.5, explanation_text, fontsize=9, family='monospace', 
        verticalalignment='center', horizontalalignment='center',
        transform=ax.transAxes)

plt.savefig(os.path.join(OUT_PLOT_DIR, "example_explanation.jpg"), dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: example_explanation.jpg")

# ============================================================================
# PRINT FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(f"""
Two funds: {fund1} vs {fund2}
Number of years: 27

Regular Hamming Distance:     {total_regular} years different
Weighted Hamming Distance:    {total_weighted}
Average |i-j| when different: {total_weighted/total_regular:.2f} clusters apart

Key Insight:
  Regular Hamming says they differ in {total_regular} years.
  Weighted Hamming shows that when they differ, they differ by
  an average of {total_weighted/total_regular:.2f} cluster levels.
  
  This captures the reviewer's concern that C5 vs C6 should not be
  treated the same as C1 vs C10.
""")
print("="*70)
print(f"All plots saved to: {OUT_PLOT_DIR}")
print("="*70)