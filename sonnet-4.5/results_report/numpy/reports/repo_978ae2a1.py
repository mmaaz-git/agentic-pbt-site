import numpy as np

A = np.array([[1.69764296e-127, 1.69764296e-127],
              [1.0, 1.0]])

eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:")
print(eigenvectors)
print()

v0 = eigenvectors[:, 0]
lam0 = eigenvalues[0]

Av = A @ v0
lam_v = lam0 * v0

print("Eigenpair 0 verification:")
print(f"  A @ v = {Av}")
print(f"  λ * v = {lam_v}")
print(f"  Match: {np.allclose(Av, lam_v)}")
print(f"  Error: {np.max(np.abs(Av - lam_v))}")