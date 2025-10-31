import numpy as np
from scipy.differentiate import derivative

def f(x_val):
    return x_val ** 2

x = 1.0

for step_factor in [1.0, 1.0003, 1.001, 1.01, 1.1]:
    try:
        res = derivative(f, x, step_factor=step_factor, maxiter=2)
        print(f"step_factor={step_factor:.4f}: SUCCESS, df={res.df:.6f}")
    except np.linalg.LinAlgError as e:
        print(f"step_factor={step_factor:.4f}: CRASH - {e}")