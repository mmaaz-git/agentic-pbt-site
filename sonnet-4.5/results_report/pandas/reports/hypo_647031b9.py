from hypothesis import given, strategies as st
from pandas.arrays import SparseArray
import numpy as np

@given(st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=50))
def test_argmin_argmax_consistent_with_dense(data):
    sparse = SparseArray(data)
    dense = sparse.to_dense()

    assert sparse.argmin() == np.argmin(dense)
    assert sparse.argmax() == np.argmax(dense)

if __name__ == "__main__":
    test_argmin_argmax_consistent_with_dense()