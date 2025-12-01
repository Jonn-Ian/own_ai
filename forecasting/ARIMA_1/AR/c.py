import pandas as pd
import numpy as np

path_file = r"E:\Git\my_repos\own_ai\forecasting\csv\cleansed.csv"
df = pd.read_csv(path_file, header=0).sort_values(["year", "month"])

# annual mean revenue
result = df.groupby("year")["revenue"].mean().reset_index()

arr = []
for i in range(len(result) - 1):
    current_year = result["year"].iloc[i + 1]
    previous_year = result["year"].iloc[i]
    current_revenue = result["revenue"].iloc[i + 1]
    previous_revenue = result["revenue"].iloc[i]

    diff = current_revenue - previous_revenue
    arr.append(round(diff))
    print(f"{current_year} - {previous_year} = {diff}")

# average annual change
c = np.mean(arr)