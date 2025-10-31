import numpy as np

# Create a 2D matrix
m = np.matrix([[1, 2, 3], [4, 5, 6]])
print(f"Original: shape={m.shape}, ndim={m.ndim}, type={type(m)}")

# Use np.newaxis indexing to create a 3D matrix (this should not be allowed!)
result = m[:, np.newaxis, :]
print(f"After indexing: shape={result.shape}, ndim={result.ndim}, type={type(result)}")
print(f"Is matrix: {isinstance(result, np.matrix)}")

# Check that we indeed have a 3D matrix object
assert result.ndim == 3
assert isinstance(result, np.matrix)
print("\nBug confirmed: Created a 3D matrix object!")

# Try to compute the inverse of the 3D matrix (this should fail)
try:
    inverse = result.I
    print(f"Inverse computed: {inverse}")
except Exception as e:
    print(f"\nAttempting to compute inverse fails with: {type(e).__name__}: {e}")