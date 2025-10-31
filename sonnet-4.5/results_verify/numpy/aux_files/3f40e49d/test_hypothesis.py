import numpy as np
from hypothesis import given, strategies as st


@given(
    st.integers(min_value=1, max_value=5),
    st.integers(min_value=1, max_value=5)
)
def test_bmat_gdict_without_ldict(rows, cols):
    A = np.matrix(np.ones((rows, cols)))
    result = np.bmat('A', ldict=None, gdict={'A': A})
    assert np.array_equal(result, A)

if __name__ == "__main__":
    test_bmat_gdict_without_ldict()