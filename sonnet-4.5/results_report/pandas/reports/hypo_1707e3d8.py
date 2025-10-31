import numpy as np
import pandas.core.arrays.sparse as sparse
from hypothesis import given, strategies as st, settings

@given(
    st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=100),
    st.integers(min_value=-10, max_value=10)
)
@settings(max_examples=200)
def test_nonzero_equivalence(data, fill_value):
    """nonzero should match dense nonzero"""
    arr = sparse.SparseArray(data, fill_value=fill_value)

    sparse_nonzero = arr.nonzero()
    dense_nonzero = arr.to_dense().nonzero()

    for s, d in zip(sparse_nonzero, dense_nonzero):
        np.testing.assert_array_equal(
            s, d,
            err_msg=f"nonzero() mismatch for data={data}, fill_value={fill_value}"
        )

if __name__ == "__main__":
    test_nonzero_equivalence()