import numpy as np
from pandas.arrays import SparseArray
from hypothesis import given, strategies as st, settings


@given(
    st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=50),
    st.sampled_from([np.int32, np.int64, np.float32, np.float64, 'int32', 'int64', 'float32', 'float64'])
)
@settings(max_examples=100)
def test_astype_always_returns_sparse_array(data, dtype):
    """
    Property: According to the docstring, astype should ALWAYS return a SparseArray.

    From the docstring:
    "The output will always be a SparseArray. To convert to a dense
    ndarray with a certain dtype, use :meth:`numpy.asarray`."
    """
    arr = np.array(data)
    sparse = SparseArray(arr)

    result = sparse.astype(dtype)

    assert isinstance(result, SparseArray), \
        f"astype({dtype}) returned {type(result)}, but docstring says it should ALWAYS return SparseArray"

if __name__ == "__main__":
    test_astype_always_returns_sparse_array()