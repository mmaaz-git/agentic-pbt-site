import numpy as np
from pandas.arrays import SparseArray
from hypothesis import given, strategies as st

@given(
    st.lists(st.integers(min_value=-10, max_value=10), min_size=1, max_size=20),
    st.integers(min_value=-10, max_value=10)
)
def test_nonzero_matches_dense(data, fill_value):
    arr = SparseArray(data, fill_value=fill_value)
    sparse_result = arr.nonzero()[0]
    dense_result = arr.to_dense().nonzero()[0]

    assert np.array_equal(sparse_result, dense_result), \
        f"sparse.nonzero() != to_dense().nonzero() for data={data}, fill_value={fill_value}"

if __name__ == "__main__":
    test_nonzero_matches_dense()