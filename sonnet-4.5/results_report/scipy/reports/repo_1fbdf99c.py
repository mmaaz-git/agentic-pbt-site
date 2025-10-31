import numpy as np
from scipy.differentiate import derivative

# Reproduce the bug with step_factor=1.0
res = derivative(np.exp, 1.5, step_factor=1.0)
print(res)