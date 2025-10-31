import numpy as np
from scipy.differentiate import derivative

# Attempt to calculate derivative with step_factor=0
# This should raise a ValueError but instead causes division by zero
result = derivative(np.sin, 1.0, step_factor=0.0)
print(f"Result: {result}")
print(f"Derivative value: {result.df}")
print(f"Success: {result.success}")
print(f"Status: {result.status}")