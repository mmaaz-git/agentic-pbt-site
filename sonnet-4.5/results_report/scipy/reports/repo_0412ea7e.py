import numpy as np
from scipy.differentiate import jacobian

# Simple test case: linear function f(x) = Ax
A = np.array([[1.0, 2.0],
              [3.0, 5.0]])

def f(x):
    return A @ x

x = np.array([1.0, 2.0])
result = jacobian(f, x)

print("Matrix A:")
print(A)
print("\nComputed Jacobian:")
print(result.df)
print("\nExpected (A):")
print(A)
print("\nActual result (A.T):")
print(A.T)
print("\nMatch with A.T:", np.allclose(result.df, A.T))
print("\nMatch with A:", np.allclose(result.df, A))

# Verify the mathematical definition
print("\n--- Mathematical Verification ---")
print("For linear function f(x) = Ax:")
print("Jacobian J[i,j] should equal A[i,j] = ∂f_i/∂x_j")
print("\nChecking elements:")
for i in range(2):
    for j in range(2):
        print(f"J[{i},{j}] = {result.df[i,j]:.1f}, A[{i},{j}] = {A[i,j]:.1f}, A.T[{i},{j}] = {A.T[i,j]:.1f}")