import warnings
import numpy as np
from numpy import matrix

warnings.filterwarnings('ignore', category=PendingDeprecationWarning)

print("Testing NumPy's singularity detection threshold")
print("=" * 60)

# Test various matrices with determinants close to zero
test_values = [1.56673884, 1.98828125, 2.0, 1.0, 0.5, 0.25]

for val in test_values:
    m = matrix([[val, val], [val, val]])
    det = np.linalg.det(m)
    cond = np.linalg.cond(m)

    print(f"\nValue: {val}")
    print(f"  Determinant: {det:.2e}")
    print(f"  Condition number: {cond:.2e}")

    try:
        inv = np.linalg.inv(m)
        print(f"  np.linalg.inv: Success (no error)")
        # Check if result is actually valid
        result = inv @ m
        is_identity = np.allclose(result, np.eye(2), atol=1e-8)
        print(f"  inv @ m is identity? {is_identity}")
    except np.linalg.LinAlgError as e:
        print(f"  np.linalg.inv: LinAlgError - {e}")

# Let's see what the documentation says about numerical precision
print("\n" + "=" * 60)
print("Testing with tiny perturbation to break singularity")
print("=" * 60)

# Perfectly singular
m_singular = matrix([[1.98828125, 1.98828125],
                     [1.98828125, 1.98828125]])

# Add tiny perturbation
epsilon = 1e-15
m_perturbed = matrix([[1.98828125, 1.98828125],
                      [1.98828125, 1.98828125 + epsilon]])

print(f"Singular matrix det: {np.linalg.det(m_singular):.2e}")
print(f"Perturbed matrix det: {np.linalg.det(m_perturbed):.2e}")

try:
    inv_s = np.linalg.inv(m_singular)
    print("Singular matrix: inverse computed")
    print(f"  inv @ m is identity? {np.allclose(inv_s @ m_singular, np.eye(2))}")
except np.linalg.LinAlgError:
    print("Singular matrix: LinAlgError raised")

try:
    inv_p = np.linalg.inv(m_perturbed)
    print("Perturbed matrix: inverse computed")
    print(f"  inv @ m is identity? {np.allclose(inv_p @ m_perturbed, np.eye(2))}")
except np.linalg.LinAlgError:
    print("Perturbed matrix: LinAlgError raised")