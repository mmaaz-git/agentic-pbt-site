import numpy as np
import numpy.linalg as la
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays

# First, let's reproduce the exact issue
print("=" * 60)
print("REPRODUCING THE REPORTED BUG")
print("=" * 60)

a = np.array([[0.00000000e+00, 1.17549435e-38, 0.00000000e+00],
              [1.00000000e+00, 1.00000000e+00, 0.00000000e+00],
              [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])

print("Matrix A:")
print(a)
print()

# Note: numpy.linalg.eig returns a tuple (eigenvalues, eigenvectors), not a namedtuple with attributes
eigenvalues, eigenvectors = la.eig(a)

print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")
print()

# Check each eigenpair
for i in range(len(eigenvalues)):
    lam = eigenvalues[i]
    v = eigenvectors[:, i]

    lhs = a @ v
    rhs = lam * v

    print(f"Eigenpair {i}:")
    print(f"  Eigenvalue: {lam}")
    print(f"  Eigenvector: {v}")
    print(f"  A @ v = {lhs}")
    print(f"  lambda * v = {rhs}")
    print(f"  Equal (default tolerance)? {np.allclose(lhs, rhs)}")
    print(f"  Equal (rtol=1e-4, atol=1e-7)? {np.allclose(lhs, rhs, rtol=1e-4, atol=1e-7)}")
    print(f"  Difference: {lhs - rhs}")
    print(f"  Max absolute difference: {np.max(np.abs(lhs - rhs))}")
    print()

# Let's also check the condition number and other properties of the matrix
print("=" * 60)
print("MATRIX ANALYSIS")
print("=" * 60)
print(f"Matrix condition number: {np.linalg.cond(a)}")
print(f"Matrix rank: {np.linalg.matrix_rank(a)}")
print(f"Matrix determinant: {np.linalg.det(a)}")

# Let's see what the actual eigenvalues should be analytically
print("\n" + "=" * 60)
print("ANALYTICAL CHECK")
print("=" * 60)

# For this matrix, we can compute eigenvalues analytically
# The characteristic polynomial is det(A - λI) = 0
print("The matrix has structure:")
print("[[0, ε, 0],")
print(" [1, 1, 0],")
print(" [0, 0, 0]]")
print(f"where ε = {1.17549435e-38}")

# Let's verify the actual eigenvector for eigenvalue 0
# If λ = 0, then A @ v = 0
# So we need to find the null space of A
print("\nNull space of A (eigenvectors for λ=0):")
from scipy.linalg import null_space
null_vecs = null_space(a)
print(f"Null space vectors:\n{null_vecs}")

# Verify these are actually eigenvectors for λ=0
if null_vecs.size > 0:
    for j in range(null_vecs.shape[1]):
        v_null = null_vecs[:, j]
        result = a @ v_null
        print(f"A @ null_vector_{j} = {result}")
        print(f"Is close to zero? {np.allclose(result, 0)}")