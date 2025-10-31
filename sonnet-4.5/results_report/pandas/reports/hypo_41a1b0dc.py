from hypothesis import given, strategies as st, settings
import numpy as np
from pandas.arrays import SparseArray

@given(
    st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=50),
    st.integers(min_value=-100, max_value=100),
)
@settings(max_examples=100)
def test_sparse_array_argmin_argmax_match_dense(data, fill_value):
    """
    Property: argmin() and argmax() should match dense array
    Evidence: _argmin_argmax method should find correct positions
    """
    sparse = SparseArray(data, fill_value=fill_value)
    dense = np.array(data)

    assert sparse.argmin() == dense.argmin()
    assert sparse.argmax() == dense.argmax()

if __name__ == "__main__":
    # Run the test
    test_sparse_array_argmin_argmax_match_dense()