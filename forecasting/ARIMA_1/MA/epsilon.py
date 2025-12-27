import pandas as pd
import numpy as np

path_file = r"E:\Git\my_repos\own_ai\forecasting\csv\cleansed.csv"
df = pd.read_csv(path_file).sort_values("year")
y = df.groupby("year")["revenue"].mean().to_numpy()

# AR(1) phi
y_t = y[1:]
y_tm1 = y[:-1]
phi = np.sum(y_t * y_tm1) / np.sum(y_tm1**2)

epsilon = y_t - phi * y_tm1
