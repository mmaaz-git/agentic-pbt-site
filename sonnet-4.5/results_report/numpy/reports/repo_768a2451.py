import numpy as np
import numpy.linalg as la

# The problematic matrix with extreme value
a = np.array([[0.00000000e+00, 1.17549435e-38, 0.00000000e+00],
              [1.00000000e+00, 1.00000000e+00, 0.00000000e+00],
              [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])

print("Matrix A:")
print(a)
print()

# Compute eigenvalues and eigenvectors
result = la.eig(a)
eigenvalues = result.eigenvalues
eigenvectors = result.eigenvectors

print("Eigenvalues:")
print(eigenvalues)
print()

print("Eigenvectors (columns):")
print(eigenvectors)
print()

# Check each eigenpair
for i in range(len(eigenvalues)):
    lambda_i = eigenvalues[i]
    v_i = eigenvectors[:, i]

    print(f"Eigenpair {i}:")
    print(f"  Eigenvalue λ = {lambda_i}")
    print(f"  Eigenvector v = {v_i}")

    # Check if A @ v = λ * v
    lhs = a @ v_i
    rhs = lambda_i * v_i

    print(f"  A @ v = {lhs}")
    print(f"  λ * v = {rhs}")

    # Check if they are equal within reasonable tolerance
    is_equal = np.allclose(lhs, rhs, rtol=1e-4, atol=1e-7)
    print(f"  A @ v ≈ λ * v? {is_equal}")

    if not is_equal:
        error = np.linalg.norm(lhs - rhs)
        print(f"  Error magnitude: {error}")
    print()