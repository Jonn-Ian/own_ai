import pandas as pd
import numpy as np
from c import annual

path_file = r"E:\Git\my_repos\own_ai\forecasting\csv\cleansed.csv"

x1 = annual["x1"]
x2 = annual["x2"]
y = annual["y"]

df = pd.read_csv(path_file, header=0)

# sort the data by year
df = df.sort_values(by="year", ascending=True)

# group the data by year
results = df.groupby("year")["revenue"].mean().reset_index()

# x1 epsilon
x1_epsilon = x1 - (results.shift(0) + x1)

# x2 epsilon
x2_epsilon = x2 - (results.shift(1) + x2)

epsilon = df["revenue"].dropna()