import numpy as np
import scipy.sparse as sp

print("Testing default dtypes for CSR array construction")
print("=" * 50)

# From coordinate format (COO)
rows = [0, 1, 1]
cols = [0, 0, 1]
data = [1.0, 0.5, 2.0]
A1 = sp.csr_array((data, (rows, cols)), shape=(2, 2))
print(f"From COO lists:")
print(f"  indices dtype: {A1.indices.dtype}")
print(f"  indptr dtype: {A1.indptr.dtype}")

# From numpy arrays with explicit int32
rows_32 = np.array([0, 1, 1], dtype=np.int32)
cols_32 = np.array([0, 0, 1], dtype=np.int32)
data_f = np.array([1.0, 0.5, 2.0])
A2 = sp.csr_array((data_f, (rows_32, cols_32)), shape=(2, 2))
print(f"\nFrom numpy arrays with int32:")
print(f"  indices dtype: {A2.indices.dtype}")
print(f"  indptr dtype: {A2.indptr.dtype}")

# From numpy arrays with explicit int64
rows_64 = np.array([0, 1, 1], dtype=np.int64)
cols_64 = np.array([0, 0, 1], dtype=np.int64)
A3 = sp.csr_array((data_f, (rows_64, cols_64)), shape=(2, 2))
print(f"\nFrom numpy arrays with int64:")
print(f"  indices dtype: {A3.indices.dtype}")
print(f"  indptr dtype: {A3.indptr.dtype}")

# From direct CSR format specification
indices = np.array([0, 0, 1], dtype=np.int32)
indptr = np.array([0, 1, 3], dtype=np.int32)
data = np.array([1.0, 0.5, 2.0])
A4 = sp.csr_array((data, indices, indptr), shape=(2, 2))
print(f"\nFrom direct CSR with int32:")
print(f"  indices dtype: {A4.indices.dtype}")
print(f"  indptr dtype: {A4.indptr.dtype}")

# Test other scipy.sparse functions with int64
print("\n" + "=" * 50)
print("Testing other scipy.sparse functions with int64 indices:")
from scipy.sparse.linalg import spsolve, splu
A_test = sp.csr_array((data, (rows, cols)), shape=(2, 2))
b = np.array([1.0, 2.0])

try:
    x = spsolve(A_test, b)
    print(f"spsolve works with int64: x = {x}")
except Exception as e:
    print(f"spsolve failed: {e}")

try:
    lu = splu(A_test)
    print(f"splu works with int64: LU factorization successful")
except Exception as e:
    print(f"splu failed: {e}")