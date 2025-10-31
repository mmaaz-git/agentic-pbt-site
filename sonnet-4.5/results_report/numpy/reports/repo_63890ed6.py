import numpy as np
from numpy.matrixlib import bmat, matrix

# Create a simple matrix
X = matrix([[1, 2]])

# Try to use bmat with gdict parameter but without ldict
# According to the function signature, ldict has a default value of None
# so this should work
try:
    result = bmat('X', gdict={'X': X})
    print(f"Success: result = {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()