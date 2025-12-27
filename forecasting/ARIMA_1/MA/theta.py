import numpy as np
from .epsilon import epsilon

epsilon_t = epsilon[1:]
epsilon_tm1 = epsilon[:-1]
theta = np.corrcoef(epsilon_t, epsilon_tm1)[0,1]

