import numpy as np
from numpy import matrix

# Create a simple 2x2 matrix
m = matrix([[1, 2], [3, 4]])
print("Matrix m:")
print(m)
print()

# Create an output matrix
out = matrix([[0.0]])
print("Output matrix:")
print(out)
print()

# Try to call ptp with the out parameter
try:
    print("Calling m.ptp(axis=None, out=out)...")
    result = m.ptp(axis=None, out=out)
    print("Result:")
    print(result)
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")