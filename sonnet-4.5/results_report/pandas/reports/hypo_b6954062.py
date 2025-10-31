from pandas.arrays import SparseArray
from hypothesis import given, strategies as st
import numpy as np

@given(st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=50))
def test_sparse_array_astype_roundtrip(values):
    arr = SparseArray(values, dtype=np.int64)
    arr_float = arr.astype(np.float64)
    arr_int = arr_float.astype(np.int64)
    assert np.array_equal(arr.to_dense(), arr_int.to_dense()), \
        f"astype roundtrip failed: {arr.to_dense()} != {arr_int.to_dense()}"

if __name__ == "__main__":
    test_sparse_array_astype_roundtrip()