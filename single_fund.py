import yfinance as yf

xlk = yf.download("XLK", start="2005-01-01", end="2025-01-01")
returns = xlk['Adj Close'].pct_change().dropna()

# Save or use in model
returns.to_csv("xlk_returns.csv")

