import pandas as pd
import numpy as np

path_file = r"E:\Git\my_repos\own_ai\forecasting\csv\cleansed.csv"
df = pd.read_csv(path_file, header=0)

# Sort by year and month
df = df.sort_values(by=["year", "month"], ascending=True)

# Group by year and calculate mean revenue
result = df.groupby("year")["revenue"].mean().reset_index()
annual = result.sort_values(by=["year"], ascending=True)

# move the revenue data by 1
annual["x1"] = annual["revenue"].shift(1)

# move the revenue data by 2
annual["x2"] = annual["revenue"].shift(2)

# the target revenue
annual["y"] = annual['revenue']
annual = annual.dropna()

print(annual)