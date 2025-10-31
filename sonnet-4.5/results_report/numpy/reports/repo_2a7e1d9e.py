import numpy as np

# Create a simple matrix
A = np.matrix([[1, 2], [3, 4]])

# Try to use bmat with gdict but no ldict (ldict=None)
# According to the documentation, both are optional parameters
try:
    result = np.bmat('A', ldict=None, gdict={'A': A})
    print(f"Success: Result is\n{result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()