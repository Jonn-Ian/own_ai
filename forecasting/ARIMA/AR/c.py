import pandas as pd
import numpy as np

path_file = r"E:\Git\my_repos\own_ai\forecasting\csv\cleansed.csv"
df = pd.read_csv(path_file, header=0)

# Sort by year and month
df = df.sort_values(by=["year", "month"], ascending=True)

# Group by year and calculate mean revenue
result = df.groupby("year")["revenue"].mean().reset_index()
result = result.sort_values(by=["year"], ascending=True)

arr = []

for i in range(len(result) - 1):
    current_year = result["year"].iloc[i + 1]
    previous_year = result["year"].iloc[i]

    current_revenue = result["revenue"].iloc[i + 1]
    previous_revenue = result["revenue"].iloc[i]

    diff = current_revenue - previous_revenue
    arr.append(diff)

    print(f"{current_year} - {previous_year} = {diff}")

# This is the final c
c = np.mean(arr)
print(f"\nAverage annual change (c): {c}")