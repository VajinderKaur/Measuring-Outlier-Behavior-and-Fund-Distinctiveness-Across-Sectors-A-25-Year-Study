import pandas as pd, glob

for f in glob.glob("../funds/*/*.csv"):
    df = pd.read_csv(f)
    df = df[df["Date"] != "2026-01-01"]
    df.to_csv(f, index=False)