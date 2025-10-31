import numpy as np
from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as npst


def finite_float_matrix(shape):
    return npst.arrays(
        dtype=np.float64,
        shape=shape,
        elements=st.floats(
            min_value=-1e6,
            max_value=1e6,
            allow_nan=False,
            allow_infinity=False
        )
    )


@given(finite_float_matrix((3, 3)))
@settings(max_examples=300)
def test_det_transpose_invariant(A):
    det_A = np.linalg.det(A)
    det_AT = np.linalg.det(A.T)
    assert np.isclose(det_A, det_AT, rtol=1e-9, atol=1e-12)


if __name__ == "__main__":
    # Run the test
    test_det_transpose_invariant()