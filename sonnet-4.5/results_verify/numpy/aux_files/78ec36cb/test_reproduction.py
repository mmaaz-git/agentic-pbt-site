import numpy as np
import numpy.linalg as LA

print("=" * 70)
print("Testing numpy.linalg.eig with very small values")
print("=" * 70)

# Test case from bug report
A = np.array([[0.0, 1.52474291e-300],
              [1.0, 1.0]])

print("\nTest Matrix A:")
print(A)
print(f"Matrix condition number: {np.linalg.cond(A)}")

# Note: numpy.linalg.eig returns a tuple, not a namedtuple with attributes
w, v = LA.eig(A)
eigenvalues = w
eigenvectors = v

print(f"\nEigenvalues: {eigenvalues}")
print("Eigenvectors:")
print(eigenvectors)

print("\n" + "=" * 50)
print("Verifying eigenvalue equation: A @ v = λ * v")
print("=" * 50)

for i in range(len(eigenvalues)):
    v_i = eigenvectors[:, i]
    lam = eigenvalues[i]

    Av = A @ v_i
    lam_v = lam * v_i

    error = np.linalg.norm(Av - lam_v)

    print(f"\nEigenvalue {i}: λ = {lam}")
    print(f"  Eigenvector v = {v_i}")
    print(f"  A @ v = {Av}")
    print(f"  λ * v = {lam_v}")
    print(f"  Error: ||A @ v - λ * v|| = {error:.15e}")
    print(f"  Valid (rtol=1e-5, atol=1e-7)? {np.allclose(Av, lam_v, rtol=1e-5, atol=1e-7)}")

# Test additional cases mentioned
print("\n" + "=" * 70)
print("Testing additional failing cases")
print("=" * 70)

test_cases = [
    np.array([[0.0, 1e-100], [1.0, 1.0]]),
    np.array([[1e-200, 1e-200], [1.0, 1.0]]),
    np.array([[0.0, 0.0], [1.0, 1.0]])  # Case with exact zeros
]

for idx, test_matrix in enumerate(test_cases):
    print(f"\n--- Test case {idx + 1} ---")
    print(f"Matrix:\n{test_matrix}")

    w, v = LA.eig(test_matrix)

    max_error = 0
    for i in range(len(w)):
        v_i = v[:, i]
        lam = w[i]
        Av = test_matrix @ v_i
        lam_v = lam * v_i
        error = np.linalg.norm(Av - lam_v)
        max_error = max(max_error, error)

    print(f"Max eigenvalue equation error: {max_error:.15e}")

# Let's also verify the mathematical correctness manually
print("\n" + "=" * 70)
print("Manual verification of eigenvalue calculation")
print("=" * 70)

A = np.array([[0.0, 1.52474291e-300],
              [1.0, 1.0]])

# Calculate eigenvalues manually using characteristic polynomial
# det(A - λI) = 0
# For 2x2 matrix [[a,b],[c,d]], eigenvalues satisfy:
# λ² - (a+d)λ + (ad-bc) = 0

a, b = A[0, 0], A[0, 1]
c, d = A[1, 0], A[1, 1]

trace = a + d  # 0 + 1 = 1
det = a*d - b*c  # 0*1 - 1.52474291e-300*1 ≈ -1.52474291e-300 ≈ 0

print(f"Matrix trace (a+d): {trace}")
print(f"Matrix determinant (ad-bc): {det}")
print(f"Characteristic polynomial: λ² - {trace}λ + {det}")

# Solve λ² - λ + 0 = 0
# λ(λ - 1) = 0
# λ = 0 or λ = 1

print(f"Expected eigenvalues from manual calculation: 0 and 1")
print(f"NumPy computed eigenvalues: {LA.eig(A)[0]}")

# For eigenvalue λ=0, find eigenvector by solving (A - 0*I)v = 0
# This gives us Av = 0
print("\nFor λ=0, solving Av = 0:")
print("[[0, 1.52474291e-300], [1, 1]] * [v1, v2] = [0, 0]")
print("This gives: 1.52474291e-300*v2 = 0 and v1 + v2 = 0")
print("Solution: v1 = -v2, but with such small coefficient, numerical issues arise")

# Check what happens with truly zero matrix element
A_zero = np.array([[0.0, 0.0],
                   [1.0, 1.0]])
w_zero, v_zero = LA.eig(A_zero)
print(f"\nWith exact zero: eigenvalues = {w_zero}")
print(f"With 1.52e-300: eigenvalues = {LA.eig(A)[0]}")

# Verify the exact zero case
print("\nVerifying exact zero case:")
for i in range(len(w_zero)):
    v_i = v_zero[:, i]
    lam = w_zero[i]
    Av = A_zero @ v_i
    lam_v = lam * v_i
    error = np.linalg.norm(Av - lam_v)
    print(f"  Eigenvalue {i}: error = {error:.15e}")