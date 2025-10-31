import numpy as np
import scipy.linalg as la

A = np.array([[2.22507386e-311, 2.22507386e-311],
              [2.22507386e-311, 2.22507386e-311]])

print(f"Matrix A:\n{A}")
print(f"Rank: {np.linalg.matrix_rank(A)}")
print(f"Determinant: {la.det(A)}")

try:
    A_inv = la.inv(A)
    print(f"inv(A):\n{A_inv}")
    print(f"Contains NaN: {np.any(np.isnan(A_inv))}")
    print(f"Contains Inf: {np.any(np.isinf(A_inv))}")
except la.LinAlgError as e:
    print(f"LinAlgError raised: {e}")