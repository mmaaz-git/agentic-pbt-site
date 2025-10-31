import numpy as np

# Test with various small values to understand the failure threshold
test_values = [1e-10, 1e-20, 1e-30, 1e-50, 1e-100, 1e-150, 1e-200, 1e-250, 1e-300]

print("Testing different magnitude values:")
print("="*60)

for val in test_values:
    A = np.array([[val, val], [1.0, 1.0]])

    eigenvalues, eigenvectors = np.linalg.eig(A)

    # Check first eigenpair
    v0 = eigenvectors[:, 0]
    lam0 = eigenvalues[0]
    Av = A @ v0
    lam_v = lam0 * v0
    error = np.max(np.abs(Av - lam_v))

    print(f"Value: {val:.2e}")
    print(f"  Eigenvalues: {eigenvalues}")
    print(f"  Error: {error:.2e}")
    print(f"  Fails: {error > 1e-9}")
    print()

# Also test the matrix rank
print("\nMatrix properties:")
A = np.array([[1.69764296e-127, 1.69764296e-127], [1.0, 1.0]])
print(f"Rank: {np.linalg.matrix_rank(A)}")
print(f"Determinant: {np.linalg.det(A)}")
print(f"Condition number: {np.linalg.cond(A)}")