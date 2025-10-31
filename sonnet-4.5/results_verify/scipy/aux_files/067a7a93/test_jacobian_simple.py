import numpy as np
from scipy.differentiate import jacobian

A = np.array([[1.0, 2.0, 3.0],
              [0.0, 4.0, 5.0],
              [0.0, 0.0, 6.0]])

def f(xi):
    return A @ xi

x = np.zeros(3)

res = jacobian(f, x)

print("For f(x) = Ax, the Jacobian should be A:")
print(A)
print("\nBut scipy returns:")
print(res.df)
print("\nA.T (transpose of A):")
print(A.T)
print("\nVerify: res.df == A.T:", np.allclose(res.df, A.T))
print("Verify: res.df == A:", np.allclose(res.df, A))