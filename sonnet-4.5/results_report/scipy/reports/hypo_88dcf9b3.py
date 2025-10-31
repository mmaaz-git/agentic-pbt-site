import numpy as np
import scipy.linalg as la
from hypothesis import given, strategies as st, settings
from hypothesis.extra import numpy as npst


@given(npst.arrays(
    dtype=np.float64,
    shape=(2, 2),
    elements=st.floats(
        min_value=-1e100,
        max_value=1e100,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=True
    )
))
@settings(max_examples=1000)
def test_inv_singular_detection(A):
    try:
        A_inv = la.inv(A, check_finite=True)

        if np.any(np.isnan(A_inv)) or np.any(np.isinf(A_inv)):
            rank = np.linalg.matrix_rank(A)
            det = la.det(A)
            assert False, f"inv() returned NaN/Inf for singular matrix (rank={rank}, det={det})"

    except la.LinAlgError:
        pass

if __name__ == "__main__":
    test_inv_singular_detection()