import pandas as pd
import numpy as np
from .c import c

path_file = r"E:\Git\my_repos\own_ai\forecasting\csv\cleansed.csv"
df = pd.read_csv(path_file, header=0).sort_values("year")
results = df.groupby("year")["revenue"].mean().reset_index()

arr = []
for i in range(len(results) - 1):
    current_year = results.iloc[i + 1]["year"]
    current_revenue = results.iloc[i + 1]["revenue"]
    previous_year = results.iloc[i]["year"]
    previous_revenue = results.iloc[i]["revenue"]

    previous_year_epsilon = previous_revenue + c
    current_revenue_epsilon = current_revenue - previous_year_epsilon
    arr.append(current_revenue_epsilon)
    print(f"{current_year} - {previous_year} with epsilon = {current_revenue_epsilon}")

epsilon = np.mean(arr)