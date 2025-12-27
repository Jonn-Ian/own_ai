import pandas as pd
import numpy as np

path_file = r"E:\Git\my_repos\own_ai\forecasting\csv\cleansed.csv"
df = pd.read_csv(path_file).sort_values(["year", "month"])
y = df.groupby("year")["revenue"].mean().to_numpy()

# Average annual change (optional helper, not part of ARIMA equation)
c = np.mean(np.diff(y))