import numpy as np
from pandas.core.arrays.sparse import SparseArray
from hypothesis import given, strategies as st


@given(
    st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=50),
    st.integers(min_value=-100, max_value=100)
)
def test_nonzero_consistency(data, fill_value):
    sparse = SparseArray(data, fill_value=fill_value)
    dense = np.array(data)

    sparse_nonzero = sparse.nonzero()[0]
    dense_nonzero = dense.nonzero()[0]

    np.testing.assert_array_equal(sparse_nonzero, dense_nonzero)

# Run the test
if __name__ == "__main__":
    # Test with the failing example
    data = [1]
    fill_value = 1
    test_nonzero_consistency(data, fill_value)
    print("Test passed!")