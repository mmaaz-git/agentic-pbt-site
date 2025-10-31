import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as linalg

rows = [0, 1, 1]
cols = [0, 0, 1]
data = [1.0, 0.5, 2.0]
A = sp.csr_array((data, (rows, cols)), shape=(2, 2))
b = np.array([1.0, 2.0])

print(f"A.indices.dtype: {A.indices.dtype}")
print(f"A.indptr.dtype: {A.indptr.dtype}")

try:
    x = linalg.spsolve_triangular(A, b, lower=True)
    print(f"Success! x = {x}")
except TypeError as e:
    print(f"TypeError: {e}")
except Exception as e:
    print(f"Other error: {type(e).__name__}: {e}")