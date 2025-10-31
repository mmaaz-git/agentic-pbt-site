import numpy as np

# Create simple 2x2 matrices
A = np.matrix([[1, 2], [3, 4]])
B = np.matrix([[5, 6], [7, 8]])

# This should work but crashes when gdict is provided without ldict
# The function signature allows ldict=None as default, so this should be valid
print("Attempting to call np.bmat('A,B', gdict={'A': A, 'B': B})")
result = np.bmat('A,B', gdict={'A': A, 'B': B})
print("Result:", result)