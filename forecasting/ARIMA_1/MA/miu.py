import pandas as pd
import numpy as np

path_file = r"E:\Git\my_repos\own_ai\forecasting\csv\cleansed.csv"
df = pd.read_csv(path_file, header=0).sort_values("year")
df = df.groupby("year")["revenue"].mean().reset_index()

y = df["revenue"].to_numpy()
miu = np.mean(y)

print("mu (baseline average):", miu)