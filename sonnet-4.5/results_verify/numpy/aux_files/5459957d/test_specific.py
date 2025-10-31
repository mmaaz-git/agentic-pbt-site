import warnings
import numpy as np
from numpy import matrix

warnings.filterwarnings('ignore', category=PendingDeprecationWarning)

# Test the new failing case found by hypothesis
test_cases = [
    # Original bug report case
    [[1.56673884, 1.56673884], [1.56673884, 1.56673884]],
    # New failing case from hypothesis
    [[1.98828125, 1.98828125], [1.98828125, 1.98828125]],
]

for i, test_case in enumerate(test_cases):
    print(f"\n{'='*60}")
    print(f"Test case {i+1}: {test_case[0]}")
    print(f"{'='*60}")

    m = matrix(test_case)
    print(f"Matrix:\n{m}")
    print(f"Determinant: {np.linalg.det(m):.2e}")

    # Check singular values
    svd = np.linalg.svd(m, compute_uv=False)
    print(f"Singular values: {svd}")
    print(f"Condition number: {np.linalg.cond(m):.2e}")

    # Try to get inverse
    print("\nTrying m.I:")
    try:
        inv = m.I
        print(f"  Success! No exception raised")
        print(f"  m.I =\n{inv}")

        result = inv @ m
        expected = np.eye(2)
        print(f"  m.I @ m =\n{result}")
        print(f"  Is identity? {np.allclose(result, expected, atol=1e-8)}")
    except np.linalg.LinAlgError as e:
        print(f"  LinAlgError raised: {e}")

    # Try direct numpy.linalg.inv
    print("\nTrying np.linalg.inv directly:")
    try:
        inv_direct = np.linalg.inv(m)
        print(f"  Success! No exception raised")
        print(f"  inv =\n{inv_direct}")
        result = inv_direct @ m
        print(f"  inv @ m =\n{result}")
    except np.linalg.LinAlgError as e:
        print(f"  LinAlgError raised: {e}")

# Let's test with slightly perturbed values
print(f"\n{'='*60}")
print("Testing with floating point precision")
print(f"{'='*60}")

# Create an exactly singular matrix using np.float64
m_exact = matrix([[np.float64(1.98828125), np.float64(1.98828125)],
                  [np.float64(1.98828125), np.float64(1.98828125)]], dtype=np.float64)
print(f"Matrix (exact float64):\n{m_exact}")
print(f"Are rows exactly equal? {np.array_equal(m_exact[0], m_exact[1])}")
print(f"Determinant: {np.linalg.det(m_exact):.2e}")

try:
    inv = m_exact.I
    print(f"m.I succeeded!")
    print(f"m.I @ m =\n{inv @ m_exact}")
except np.linalg.LinAlgError as e:
    print(f"LinAlgError raised: {e}")