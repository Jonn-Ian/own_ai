import pandas as pd
import numpy as np
from .phi import phi

path_file = r"E:\Git\my_repos\own_ai\forecasting\csv\cleansed.csv"
df = pd.read_csv(path_file).sort_values("year")
y = df.groupby("year")["revenue"].mean().to_numpy()

# Differences series
d = np.diff(y) # get the differences

# from AR(1) fit
d_t = d[1:]
d_tm1 = d[:-1]
epsilon = d_t - phi * d_tm1



# this is what's happening here
# let's say we got a list of number like before
# [3, 2, 6, 1]

# differences = 2 - 3, 6 - 2, 1 - 6
# differences = -1, 4, -5

# latest rev = 4, -5
# prev rev = -1, 4 

# numerator = latest * prev
# numerator = 4 * -1, -5 * 4
# numerator = -4 + -20 = numerator = -24

# denominator = (prev)^2
# denominator = -1^2 + 4^2 = denominator = 1 + 16
# denominator = 17

# phi = numerator/denominator = -24/17 = 1.41
# epsilon = latest - phi * prev 