import numpy as np
import scipy.differentiate

def f(x_val):
    return x_val**2

print("Testing if initial_step=0 raises ValueError...")
try:
    result = scipy.differentiate.derivative(f, 0.0, initial_step=0)
    print(f"No error raised. Result: success={result.success}, df={result.df}, status={result.status}")
except ValueError as e:
    print(f"ValueError raised: {e}")

print("\nTesting if initial_step=-0.5 raises ValueError...")
try:
    result = scipy.differentiate.derivative(f, 0.0, initial_step=-0.5)
    print(f"No error raised. Result: success={result.success}, df={result.df}, status={result.status}")
except ValueError as e:
    print(f"ValueError raised: {e}")