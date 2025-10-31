from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.numpy import arrays
import numpy as np
import scipy.linalg

def square_matrix_strategy(n):
    return arrays(dtype=np.float64, shape=(n, n),
                  elements=st.floats(min_value=-100, max_value=100,
                                   allow_nan=False, allow_infinity=False))

@given(square_matrix_strategy(3))
@settings(max_examples=200, deadline=None)
def test_funm_consistency(A):
    try:
        norm_A = np.linalg.norm(A)
        assume(norm_A < 10)

        exp_A_funm = scipy.linalg.funm(A, np.exp)
        exp_A_direct = scipy.linalg.expm(A)

        if not np.iscomplexobj(exp_A_funm):
            assert np.allclose(exp_A_funm, exp_A_direct, rtol=1e-3, atol=1e-5), \
                f"funm(A, exp) should equal expm(A)"
    except (np.linalg.LinAlgError, ValueError, scipy.linalg.LinAlgError, OverflowError):
        assume(False)

# Run the test
if __name__ == "__main__":
    test_funm_consistency()
    print("All tests passed!")