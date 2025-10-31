import numpy as np
import scipy.linalg as la

# Using the failing example from Hypothesis
A = np.array([[5.e-324, 5.e-324],
              [5.e-324, 5.e-324]])

print(f"Matrix A:\n{A}")
print(f"Rank: {np.linalg.matrix_rank(A)}")
print(f"Determinant: {la.det(A)}")

try:
    A_inv = la.inv(A)
    print(f"inv(A):\n{A_inv}")
    print("No LinAlgError was raised!")
except la.LinAlgError as e:
    print(f"LinAlgError was raised: {e}")