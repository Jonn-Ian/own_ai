import pandas as pd
from .c import c
from .epsilon import epsilon

path_file = r"E:\Git\my_repos\own_ai\forecasting\csv\cleansed.csv"
df = pd.read_csv(path_file, header=0)

# filter only 2025 rows
df_2025 = df[df["year"] == 2025]

# mean revenue for 2025
latest_revenue = df_2025["revenue"].mean()

# forecast next year (2026)
forecasted_year = latest_revenue + c + epsilon