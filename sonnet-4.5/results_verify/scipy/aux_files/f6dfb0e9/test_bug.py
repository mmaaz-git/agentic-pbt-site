import numpy as np
from scipy.differentiate import derivative

def f(x):
    return x**2

x = 2.0

print("Testing with step_factor=1.0:")
try:
    result = derivative(f, x, step_factor=1.0, order=4, maxiter=2)
    print(f"Result: {result.df}")
except np.linalg.LinAlgError as e:
    print(f"LinAlgError raised: {e}")
    print("This is a bug - should be caught in input validation")
except Exception as e:
    print(f"Other exception raised: {type(e).__name__}: {e}")