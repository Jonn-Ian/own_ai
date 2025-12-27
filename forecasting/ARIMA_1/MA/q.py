import numpy as np
from .epsilon import epsilon
from .miu import miu
from .theta import theta
from .yesterday_error import yesterday_error

# Simplified MA forecast component
q = miu + epsilon[-1] + theta * yesterday_error[-1]
