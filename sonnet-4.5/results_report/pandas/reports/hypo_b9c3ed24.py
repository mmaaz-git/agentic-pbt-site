from hypothesis import given, strategies as st
from pandas.arrays import SparseArray
import numpy as np

@given(st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=100))
def test_astype_preserves_values(data):
    sparse = SparseArray(data, dtype=np.int64)
    sparse_float = sparse.astype(np.float64)

    assert isinstance(sparse_float, SparseArray), f"Expected SparseArray, got {type(sparse_float)}"
    assert np.array_equal(sparse.to_dense(), sparse_float.to_dense())

if __name__ == "__main__":
    test_astype_preserves_values()