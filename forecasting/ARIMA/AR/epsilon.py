import pandas as pd
import numpy as np
from c import c

path_file = r"E:\Git\my_repos\own_ai\forecasting\csv\cleansed.csv"

df = pd.read_csv(path_file, header=0)

# sort the data by year
df = df.sort_values(by="year", ascending=True)
# group the data by year
results = df.groupby("year")["revenue"].mean().reset_index()

arr = []

for i in range(len(results) - 1):

    # current year and revenue
    current_year = results.iloc[i + 1]["year"]
    current_revenue = results.iloc[i + 1]["revenue"]

    # previous year, revenue, previous year's revenue with epsilon
    previous_year = results.iloc[i]["year"]
    previous_revenue = results.iloc[i]["revenue"]
    previous_year_epsilon = previous_revenue + c

    current_revenue_epsilon = current_revenue - previous_year_epsilon
    arr.append(current_revenue_epsilon)
    print(f"{current_year} - {previous_year} with epsilon = {current_revenue_epsilon}")

# final epsilon
epsilon = np.mean(arr)
print(f"\nAverage annual change with epsilon: {epsilon}")
