# AR/forecasted_year.py
import pandas as pd
import numpy as np
from .phi import phi

path_file = r"E:\Git\my_repos\own_ai\forecasting\csv\cleansed.csv"
df = pd.read_csv(path_file).sort_values("year")
y = df.groupby("year")["revenue"].mean().to_numpy()

d = np.diff(y)
forecast_diff = phi * d[-1]
forecasted_year = y[-1] + forecast_diff