import numpy as np
from numpy.matrixlib import bmat, matrix

# Test the minimal example from the bug report
X = matrix([[1, 2]])
print(f"Creating matrix X: {X}")
print(f"Type of X: {type(X)}")

try:
    result = bmat('X', gdict={'X': X})
    print(f"Result: {result}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")

# Test with ldict provided
try:
    result2 = bmat('X', ldict={}, gdict={'X': X})
    print(f"With empty ldict, result: {result2}")
except Exception as e:
    print(f"With ldict error: {type(e).__name__}: {e}")