from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays
import numpy as np
import scipy.sparse as sp


@st.composite
def matmul_compatible_matrices(draw, max_size=10):
    m = draw(st.integers(min_value=1, max_value=max_size))
    n = draw(st.integers(min_value=1, max_value=max_size))
    k = draw(st.integers(min_value=1, max_value=max_size))

    shape_a = (m, n)
    shape_b = (n, k)

    dense_a = draw(arrays(
        dtype=np.float64,
        shape=shape_a,
        elements=st.floats(
            min_value=-100, max_value=100,
            allow_nan=False, allow_infinity=False
        )
    ))
    dense_b = draw(arrays(
        dtype=np.float64,
        shape=shape_b,
        elements=st.floats(
            min_value=-100, max_value=100,
            allow_nan=False, allow_infinity=False
        )
    ))

    mask_a = draw(arrays(dtype=np.bool_, shape=shape_a, elements=st.booleans()))
    mask_b = draw(arrays(dtype=np.bool_, shape=shape_b, elements=st.booleans()))

    dense_a = dense_a * mask_a
    dense_b = dense_b * mask_b

    return sp.dia_array(dense_a), sp.dia_array(dense_b)


@given(matmul_compatible_matrices())
@settings(max_examples=100)
def test_matmul_matches_numpy(matrices):
    A, B = matrices

    sp_result = (A @ B).toarray()
    np_result = A.toarray() @ B.toarray()

    assert np.allclose(sp_result, np_result, rtol=1e-8, atol=1e-8)

if __name__ == "__main__":
    test_matmul_matches_numpy()