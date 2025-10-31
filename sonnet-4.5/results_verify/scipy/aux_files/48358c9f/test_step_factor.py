from scipy.differentiate import derivative
import numpy as np

# Test with step_factor=0.0 - The bug report expects this to raise ValueError
# but claims it doesn't
try:
    result = derivative(np.sin, 1.0, step_factor=0.0)
    print(f"Test failed - no ValueError was raised, got result: {result}")
except ValueError as e:
    print(f"Test passed - ValueError was raised: {e}")
except Exception as e:
    print(f"Different exception raised: {type(e).__name__}: {e}")