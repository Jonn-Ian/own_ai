import numpy as np
from .epsilon import epsilon
from .miu import miu
from .theta import theta
from .yesterday_error import yesterday_error

# Use the most recent error instead of the whole array
q = miu + epsilon[-1] + theta * yesterday_error[-1]

print("q:", q)