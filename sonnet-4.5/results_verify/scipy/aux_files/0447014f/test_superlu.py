import numpy as np
from scipy.sparse.linalg._dsolve import _superlu

# Test what types _superlu.gstrs accepts
trans = "N"
N = 2

# Prepare L matrix (lower triangular)
L_data = np.array([1.0, 0.5, 2.0])
L_indices_32 = np.array([0, 1, 1], dtype=np.int32)
L_indptr_32 = np.array([0, 1, 3], dtype=np.int32)

L_indices_64 = np.array([0, 1, 1], dtype=np.int64)
L_indptr_64 = np.array([0, 1, 3], dtype=np.int64)

# Prepare U matrix (empty for lower triangular case)
U_nnz = 0
U_data = np.array([], dtype=np.float64)
U_indices = np.array([], dtype=np.int32)
U_indptr = np.array([0, 0, 0], dtype=np.int32)

b = np.array([1.0, 2.0])

print("Testing _superlu.gstrs with int32 indices:")
try:
    x, info = _superlu.gstrs(trans,
                             N, len(L_data), L_data, L_indices_32, L_indptr_32,
                             N, U_nnz, U_data, U_indices, U_indptr,
                             b.copy())
    print(f"  Success! x = {x}, info = {info}")
except Exception as e:
    print(f"  Failed: {type(e).__name__}: {e}")

print("\nTesting _superlu.gstrs with int64 indices:")
try:
    x, info = _superlu.gstrs(trans,
                             N, len(L_data), L_data, L_indices_64, L_indptr_64,
                             N, U_nnz, U_data, U_indices, U_indptr,
                             b.copy())
    print(f"  Success! x = {x}, info = {info}")
except Exception as e:
    print(f"  Failed: {type(e).__name__}: {e}")