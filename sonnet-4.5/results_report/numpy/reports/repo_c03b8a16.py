import numpy as np
import scipy.linalg

A = np.array([[-1.0, -1e-50],
               [ 1.0, -1.0]])

print("Original matrix A:")
print(A)
print(f"Determinant: {np.linalg.det(A):.6f}")
print(f"Condition number: {np.linalg.cond(A):.2f}")

logA = scipy.linalg.logm(A)
result = scipy.linalg.expm(logA)

print("\nexpm(logm(A)):")
print(result)

print("\nExpected (original A):")
print(A)

print("\nError:")
print(f"||expm(logm(A)) - A|| = {np.linalg.norm(result - A):.2e}")