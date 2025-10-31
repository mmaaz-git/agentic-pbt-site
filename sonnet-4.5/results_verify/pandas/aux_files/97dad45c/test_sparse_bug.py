import pandas as pd
from hypothesis import given, strategies as st
from pandas.api.extensions import take


@given(
    arr=st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=100),
)
def test_take_with_sparse_array(arr):
    sparse = pd.arrays.SparseArray(arr)
    indices = [0, 0, 0]
    result = take(sparse, indices, allow_fill=False)
    assert isinstance(result, pd.arrays.SparseArray)
    assert all(r == arr[0] for r in result)

if __name__ == "__main__":
    test_take_with_sparse_array()