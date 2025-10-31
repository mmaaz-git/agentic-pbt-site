import warnings
import numpy as np
from numpy import matrix

warnings.filterwarnings('ignore', category=PendingDeprecationWarning)

# Test the specific failing case mentioned in the bug report
m = matrix([[1.56673884, 1.56673884],
            [1.56673884, 1.56673884]])

print("=" * 60)
print("Testing the specific failing case from bug report")
print("=" * 60)
print(f"Matrix:\n{m}")
print(f"Determinant: {np.linalg.det(m):.2e}")

try:
    inv = m.I
    print(f"\nm.I returned successfully (no exception raised)")
    print(f"m.I =\n{inv}")

    result = inv @ m
    expected = np.eye(2)

    print(f"\nm.I @ m =\n{result}")
    print(f"\nExpected identity matrix:\n{expected}")
    print(f"\nAre they equal with allclose? {np.allclose(result, expected, atol=1e-8)}")

    # Also test the other direction
    result2 = m @ inv
    print(f"\nm @ m.I =\n{result2}")
    print(f"Are they equal with allclose? {np.allclose(result2, expected, atol=1e-8)}")

except np.linalg.LinAlgError as e:
    print(f"LinAlgError raised: {e}")

# Test with numpy.linalg.inv directly
print("\n" + "=" * 60)
print("Testing numpy.linalg.inv directly on the same matrix")
print("=" * 60)
try:
    inv_direct = np.linalg.inv(m)
    print(f"np.linalg.inv returned successfully (no exception raised)")
    print(f"inv =\n{inv_direct}")
    print(f"inv @ m =\n{inv_direct @ m}")
except np.linalg.LinAlgError as e:
    print(f"LinAlgError raised by np.linalg.inv: {e}")

# Test with a truly singular matrix (all zeros)
print("\n" + "=" * 60)
print("Testing with zero matrix (definitely singular)")
print("=" * 60)
zero_m = matrix([[0, 0], [0, 0]])
print(f"Matrix:\n{zero_m}")
print(f"Determinant: {np.linalg.det(zero_m):.2e}")

try:
    inv_zero = zero_m.I
    print(f"zero_m.I returned successfully (no exception raised)")
    print(f"zero_m.I =\n{inv_zero}")
except np.linalg.LinAlgError as e:
    print(f"LinAlgError raised: {e}")

# Test the condition number
print("\n" + "=" * 60)
print("Checking condition number of the reported matrix")
print("=" * 60)
m_arr = np.array([[1.56673884, 1.56673884],
                  [1.56673884, 1.56673884]])
print(f"Condition number: {np.linalg.cond(m_arr)}")
print(f"Singular values: {np.linalg.svd(m_arr, compute_uv=False)}")