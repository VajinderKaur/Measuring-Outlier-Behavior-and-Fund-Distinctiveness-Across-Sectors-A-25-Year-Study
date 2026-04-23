# Mutual Fund Performance Analysis: Outlier Detection and Diversification

A comprehensive study investigating the identification and behavioral patterns of outlier mutual fund managers across multiple sectors over a 27-year period (1999–2025).

---

## Overview

This repository contains the code and data for analyzing sector-specific mutual fund performance using daily NAV returns. The analysis employs fund-year L2 (RMS) distance from market behavior, quantile tracking, and Hamming distance metrics to identify outlier funds and assess portfolio diversification across:

- Energy
- Healthcare
- Real Estate
- Technology
- Utilities

---

## Key Features

- Market Model Regression: isolates fund-specific performance using S&P 500 benchmark  
- Fund-Year L2 Framework: RMS deviation from market per fund per year  
- Quantile Classification: cross-sectional decile ranking of fund-year distances  
- Trajectory Analysis: evolution of fund behavior across time  
- Hamming Distance Analysis: behavioral divergence between funds  
- Diversification Contribution: leave-one-out impact analysis  
- Sector + Cross-Sector Comparisons

---

## Repository Structure

Finance/
├── benchmarks/            # S&P 500 benchmark data  
├── clusters/              # FUND-YEAR L2 QUANTILE DATA (FINAL)  
├── funds/                 # Raw NAV data by sector  
├── logs/                  # Processing logs  
├── plots/                 # All visualizations  
├── quantiles/             # Quantile outputs and intermediate files  
├── residuals/             # Market model residuals  
├── scripts/               # Analysis scripts  
│   ├── 01_data_collection.py  
│   ├── 02_residuals.py  
│   ├── 03_clusters.py  
│   ├── 04.1_2fundstrajectory.py  
│   ├── 04.2_trajectory.py  
│   ├── 05.1_hamming_matrix.py  
│   ├── 05.2_hamming_sector.py  
│   ├── 05.3_hamming_intersector.py  
│   └── misc/  
├── README.md  
├── requirements.txt  
└── venv/                   # Local virtual environment (not for distribution)

---

## Methodology

### Analysis Pipeline

main pipeline (conceptual):
01_data_collection.py → 02_residuals.py → 03_clusters.py → 04.* trajectory scripts → 05.* hamming scripts → plots/

---

## 1. Market Model Regression

For each fund i:

R_{i,t} = α_i + β_i R_{m,t} + ε_{i,t}

Where:
- R_{i,t}: fund return  
- R_{m,t}: market return (S&P 500)  
- ε_{i,t}: idiosyncratic residual

---

## 2. Fund-Year L2 Distance

We compute:

L2_{i,y} = sqrt( (1/T) sum_t ε_{i,t}^2 )

This gives a single distance-from-market measure per fund-year.

---

## 3. Decile Classification

Fund-year L2 distances are ranked into deciles:

- Q1 → closest to market behavior  
- Q10 → most divergent behavior

---

### Why Deciles Instead of K-Means?

We chose quantile-based classification over clustering methods like k-means for several important reasons:

1. Simplicity and Interpretability  
   Quantiles provide a direct, intuitive measure of relative performance. Each decile represents exactly 10% of the distribution, making results easy to communicate and interpret.

2. Consistent Grouping Across Time  
   Unlike k-means, which produces uneven and unstable clusters, quantiles guarantee consistent group sizes across all years and sectors.

3. Deterministic and Reproducible  
   K-means depends on initialization and can vary across runs. Quantiles are fully deterministic.

4. Direct Alignment with Research Goal  
   We are measuring distance from market behavior, not geometric similarity. Ranking residual magnitudes is therefore more appropriate than clustering in feature space.

5. Temporal Stability  
   Quantile boundaries remain consistent across time, whereas k-means clusters can drift across periods.

6. Small Sample Robustness  
   With ~10–15 funds per sector, k-means becomes unstable. Quantiles remain robust.

In short, we prioritize relative position in the distribution, not clustering structure.

---

## 4. Hamming Distance

Measures how often two funds differ in assigned regimes:

d_H(Q_i, Q_j) = (1/T) sum_t 1(q_{i,t} != q_{j,t})

---

## Outputs

Each sector produces files under:

plots/hammingdistance/sectors/<sector>/

Containing:
- heatmap.jpg
- hamming_matrix.csv
- impact_table.csv
- summary.txt

Quantile outputs are available in the `quantiles/` and `clusters/` folders; residuals are stored in `residuals/`.

---

## Key Findings (Summary)

Energy
- Moderate diversification  
- Strong commodity-driven synchronization

Healthcare
- Highest internal heterogeneity  
- Strong subsector structure

Real Estate
- Lowest diversification  
- High macro sensitivity (rates/credit)

---

## Legacy Data

The scripts/misc and other legacy folders contain:
- old daily-level residuals  
- early k-means experiments  
- exploratory plots

⚠️ Not used in final L2 framework

---

## Citation

@article{mutual_fund_outliers_2025,  
  title={Outlier Mutual Fund Managers: Identification and Behavioral Patterns Across Sectors},  
  author={Vajinder Kaur, Eugene Pinsky},  
  year={2025},  
  journal={Working Paper}  
}

---

## Contact

vajinder@bu.edu  
epinsky@bu.edu
