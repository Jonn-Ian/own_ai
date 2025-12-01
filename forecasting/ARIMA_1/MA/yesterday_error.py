import numpy as np
from .epsilon import epsilon

yesterday_error = epsilon[:-1]

print("Yesterday's error (epsilon_tm1):", yesterday_error)