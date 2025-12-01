import pandas as pd
import numpy as np

path_file = r"E:\Git\my_repos\own_ai\forecasting\csv\cleansed.csv"
df = pd.read_csv(path_file, header=0).sort_values("year")
df = df.groupby("year")["revenue"].mean().reset_index()

y = df["revenue"].to_numpy()

# AR(1) phi
y_t   = y[1:]
y_tm1 = y[:-1]
phi = np.mean(np.divide(y_t, y_tm1, out=np.zeros_like(y_t), where=y_tm1 != 0))

# epsilon_t = actual - predicted
epsilon = y_t - phi * y_tm1

print("phi:", phi)
print("epsilon:", epsilon)