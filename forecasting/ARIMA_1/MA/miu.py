import pandas as pd
import numpy as np

path_file = r"E:\Git\my_repos\own_ai\forecasting\csv\cleansed.csv"
df = pd.read_csv(path_file).sort_values("year")
y = df.groupby("year")["revenue"].mean().to_numpy()

miu = np.mean(y)
