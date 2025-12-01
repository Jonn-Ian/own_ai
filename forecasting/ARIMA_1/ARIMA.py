import pandas as pd
import numpy as np
from AR.p import p
from I.d import d
from MA.q import q

path_file = r"E:\Git\my_repos\own_ai\forecasting\csv\cleansed.csv"

df = pd.read_csv(path_file, header=0).sort_values(by="year", ascending=True)
df = df.groupby("year")["revenue"].mean().reset_index()

# reduce d (list) and q (array) to scalars
arima = p + sum(d) + np.mean(q)

print("\nARIMA forecast:", arima)

print("\nYear-to-year revenue differences:")
for i in range(1, len(df)):
    recent_year = int(df["year"].iloc[i])
    prev_year   = int(df["year"].iloc[i-1])
    diff        = df["revenue"].iloc[i] - df["revenue"].iloc[i-1]
    print(f"{recent_year} - {prev_year} = {diff:.2f}")