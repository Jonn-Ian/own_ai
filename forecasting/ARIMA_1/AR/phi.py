import pandas as pd

path_file = r"E:\Git\my_repos\own_ai\forecasting\csv\cleansed.csv"
df = pd.read_csv(path_file, header=0).sort_values("year")
df = df.groupby("year")["revenue"].mean().reset_index()

y = df["revenue"].to_numpy()

# lagged series
y_t   = y[1:]
y_tm1 = y[:-1]

# phi estimates as ratios
phi_estimates = y_t / y_tm1
phi = phi_estimates.mean()
