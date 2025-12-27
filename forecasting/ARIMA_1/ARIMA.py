import pandas as pd
import numpy as np
from .AR.p import p
from .I.d import d

path_file = r"E:\Git\my_repos\own_ai\forecasting\csv\cleansed.csv"
df = pd.read_csv(path_file).sort_values("year")
df = df.groupby("year")["revenue"].mean().reset_index()

# ARIMA(1,1,0) forecast
print("ARIMA(1,1,0) forecast:", p)

# Print year-to-year differences
print("\nYear-to-year revenue differences:")
for i in range(1, len(df)):
    diff = df["revenue"].iloc[i] - df["revenue"].iloc[i-1]
    print(f"{df['year'].iloc[i]} - {df['year'].iloc[i-1]} = {diff:.2f}")