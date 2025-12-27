import pandas as pd
import numpy as np

path_file = r"E:\Git\my_repos\own_ai\forecasting\csv\cleansed.csv"
df = pd.read_csv(path_file).sort_values("year")
y = df.groupby("year")["revenue"].mean().to_numpy()

d = np.diff(y)
d_t = d[1:]
d_tm1 = d[:-1]
phi = np.sum(d_t * d_tm1) / np.sum(d_tm1**2)

# let's say that we have a list of number like
# [3, 2, 6, 1]

# current revenue = 2, 6, 1
# prev revenue = 3, 2, 6

# to get the percentage of increase or decrease

# numerator = current revenue * pre revenue
# numerator = 2 * 3, 6 * 2, 1 * 6
# numerator = 6, 12, 6

# denominator = previous val^2
# denominator = (3^2, 2^2, 6^2)
# denominator = 9, 4, 36

# phi = (numerator)sum / (denominator)sum
# phi = 6 + 12 + 6 / 9 + 4 + 36 = phi = 24/48
# phi = 0.49