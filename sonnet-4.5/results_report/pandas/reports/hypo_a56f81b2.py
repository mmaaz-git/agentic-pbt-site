from hypothesis import given, strategies as st, settings
import numpy as np
from pandas.core.arrays.sparse import SparseArray

@given(
    data=st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=100),
    fill_value=st.integers(min_value=-1000, max_value=1000)
)
@settings(max_examples=1000, deadline=None)
def test_sparse_array_argmax_argmin_match_dense(data, fill_value):
    """Test that SparseArray.argmax/argmin matches NumPy array behavior"""
    arr = np.array(data)
    sparse = SparseArray(arr, fill_value=fill_value)

    # Test argmax
    try:
        sparse_argmax = sparse.argmax()
        numpy_argmax = arr.argmax()
        assert sparse_argmax == numpy_argmax, f"argmax mismatch: sparse={sparse_argmax}, numpy={numpy_argmax}"
    except Exception as e:
        print(f"argmax failed with data={data}, fill_value={fill_value}")
        raise

    # Test argmin
    try:
        sparse_argmin = sparse.argmin()
        numpy_argmin = arr.argmin()
        assert sparse_argmin == numpy_argmin, f"argmin mismatch: sparse={sparse_argmin}, numpy={numpy_argmin}"
    except Exception as e:
        print(f"argmin failed with data={data}, fill_value={fill_value}")
        raise

if __name__ == "__main__":
    # Run the test
    test_sparse_array_argmax_argmin_match_dense()
    print("All tests passed!")