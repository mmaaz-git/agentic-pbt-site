import numpy as np
from scipy.differentiate import derivative

def f(x_val):
    return x_val ** 2

x = 1.5
try:
    res = derivative(f, x, step_factor=1.0, maxiter=3)
    print(f"step_factor=1.0: SUCCESS, df={res.df:.6f}")
except np.linalg.LinAlgError as e:
    print(f"step_factor=1.0: CRASH - {e}")