import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as linalg

# Test with int32 indices
rows = [0, 1, 1]
cols = [0, 0, 1]
data = [1.0, 0.5, 2.0]
A = sp.csr_array((data, (rows, cols)), shape=(2, 2))

# Convert to int32
A.indices = A.indices.astype(np.int32)
A.indptr = A.indptr.astype(np.int32)

b = np.array([1.0, 2.0])

print(f"A.indices.dtype: {A.indices.dtype}")
print(f"A.indptr.dtype: {A.indptr.dtype}")

try:
    x = linalg.spsolve_triangular(A, b, lower=True)
    print(f"Success! x = {x}")
    # Verify solution
    result = A @ x
    print(f"A @ x = {result}")
    print(f"b = {b}")
    print(f"Close? {np.allclose(result, b)}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")