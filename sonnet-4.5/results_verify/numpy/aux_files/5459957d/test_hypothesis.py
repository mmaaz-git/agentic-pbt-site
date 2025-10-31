import warnings
import numpy as np
from numpy import matrix
from hypothesis import given, strategies as st, assume, settings
import hypothesis.extra.numpy as npst

warnings.filterwarnings('ignore', category=PendingDeprecationWarning)

@given(st.integers(min_value=2, max_value=4).flatmap(
    lambda n: npst.arrays(dtype=np.float64, shape=(n, n),
                          elements=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False))))
@settings(max_examples=100)
def test_matrix_inverse_left_identity(arr):
    m = matrix(arr)
    try:
        inv = m.I
    except np.linalg.LinAlgError:
        assume(False)

    result = inv @ m
    n = m.shape[0]
    expected = np.eye(n)
    assert np.allclose(result, expected, atol=1e-8), f"m.I * m should equal identity but got {result}"

# Run the test
if __name__ == "__main__":
    print("Running property-based test...")
    test_matrix_inverse_left_identity()
    print("All tests passed!")