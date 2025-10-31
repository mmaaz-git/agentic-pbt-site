import numpy as np
from scipy.differentiate import derivative

# Test the bug: step_factor=1.0
print("Testing with step_factor=1.0:")
try:
    res = derivative(np.exp, 1.5, step_factor=1.0)
    print(f"Success: res.df = {res.df}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")