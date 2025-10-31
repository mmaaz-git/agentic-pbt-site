import numpy as np
import scipy.sparse as sp
from hypothesis import given, strategies as st, settings


@st.composite
def sparse_coo_matrix(draw):
    m = draw(st.integers(min_value=1, max_value=20))
    n = draw(st.integers(min_value=1, max_value=20))
    nnz = draw(st.integers(min_value=0, max_value=min(m * n, 50)))

    if nnz == 0:
        return sp.coo_matrix((m, n), dtype=np.float64)

    rows = draw(st.lists(st.integers(min_value=0, max_value=m-1), min_size=nnz, max_size=nnz))
    cols = draw(st.lists(st.integers(min_value=0, max_value=n-1), min_size=nnz, max_size=nnz))
    data = draw(st.lists(st.floats(min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False), min_size=nnz, max_size=nnz))

    return sp.coo_matrix((data, (rows, cols)), shape=(m, n))


@given(sparse_coo_matrix())
@settings(max_examples=100)
def test_transpose_preserves_canonical_format(A):
    A.sum_duplicates()
    assert A.has_canonical_format == True

    A_T = A.transpose()

    print(f"Test failed: A.has_canonical_format = {A.has_canonical_format}, A_T.has_canonical_format = {A_T.has_canonical_format}")
    assert A_T.has_canonical_format == True


# Run the test
try:
    test_transpose_preserves_canonical_format()
    print("Test passed!")
except AssertionError as e:
    print(f"Test failed as expected - transpose does not preserve canonical format flag")
except Exception as e:
    print(f"Test error: {e}")