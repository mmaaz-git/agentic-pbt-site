import numpy as np

# Create test matrices
A = np.matrix([[1, 2], [3, 4]])
B = np.matrix([[5, 6], [7, 8]])

# Try to call bmat with only gdict (no ldict)
print("Attempting to call np.bmat('A,B', gdict={'A': A, 'B': B})")
try:
    result = np.bmat('A,B', gdict={'A': A, 'B': B})
    print("Result:")
    print(result)
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()