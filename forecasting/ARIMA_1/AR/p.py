from .c import c
from .epsilon import epsilon
from .forecasted_year import forecasted_year
from .phi import phi

# final p calculation
p = round(c + forecasted_year + epsilon + phi)
