from hypothesis import given, strategies as st, settings, assume
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

@given(st.integers(min_value=1, max_value=10), st.random_module())
@settings(max_examples=100)
def test_inv_double_inversion(n, random):
    A_dense = np.random.rand(n, n) + np.eye(n) * 2
    A = sp.csc_array(A_dense)
    assume(np.linalg.det(A_dense) > 0.01)

    Ainv = sla.inv(A)
    Ainvinv = sla.inv(Ainv)

    result = Ainvinv.toarray()
    expected = A.toarray()
    assert np.allclose(result, expected, rtol=1e-5, atol=1e-8)

# Run the test
try:
    test_inv_double_inversion()
    print("Hypothesis test passed!")
except Exception as e:
    print(f"Hypothesis test failed: {e}")
    print(f"Error type: {type(e).__name__}")

# Test specifically with n=1
print("\nTesting specifically with n=1:")
np.random.seed(42)
A_dense = np.random.rand(1, 1) + np.eye(1) * 2
A = sp.csc_array(A_dense)
print(f"A_dense: {A_dense}")
print(f"Determinant: {np.linalg.det(A_dense)}")

try:
    Ainv = sla.inv(A)
    print(f"Ainv type: {type(Ainv)}")
    print(f"Ainv value: {Ainv}")
    Ainvinv = sla.inv(Ainv)
    print(f"Double inversion succeeded")
except Exception as e:
    print(f"Double inversion failed: {type(e).__name__}: {e}")