import numpy as np
from scipy.differentiate import derivative

def f(x_val):
    return x_val ** 2

x = 1.0

try:
    res = derivative(f, x, step_factor=1.0, maxiter=2)
    print(f"Result: {res.df}")
except np.linalg.LinAlgError as e:
    print(f"Crash: {e}")