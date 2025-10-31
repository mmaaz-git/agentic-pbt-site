import numpy as np
from pandas.arrays import SparseArray
from hypothesis import given, strategies as st
import sys
import traceback

@given(st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=100))
def test_cumsum_matches_dense(data):
    arr = SparseArray(data, fill_value=0)

    if not arr._null_fill_value:
        dense = arr.to_dense()
        sparse_cumsum = arr.cumsum().to_dense()
        dense_cumsum = dense.cumsum()
        np.testing.assert_array_equal(sparse_cumsum, dense_cumsum)

# Run the test
print("Running hypothesis test...")
try:
    test_cumsum_matches_dense()
    print("Test passed!")
except Exception as e:
    print(f"Test failed with error: {e}")
    traceback.print_exc()