import numpy as np
from scipy.differentiate import derivative

def f(x):
    return np.exp(x)

print("Testing with step_factor=1.0 exactly:")
try:
    result = derivative(f, 1.0, step_factor=1.0, order=4, maxiter=2)
    print(f"Result: {result.df}")
except np.linalg.LinAlgError as e:
    print(f"LinAlgError raised: {e}")
except ValueError as e:
    print(f"ValueError raised: {e}")