from hypothesis import given, strategies as st
import numpy as np
from pandas.core.arrays.sparse import SparseArray

@given(st.lists(
    st.floats(allow_nan=True, allow_infinity=False) |
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    min_size=1, max_size=100
))
def test_sum_matches_dense_skipna_false(data):
    arr = SparseArray(data, fill_value=0.0)

    sparse_sum = arr.sum(skipna=False)
    dense_sum = arr.to_dense().sum()

    if np.isnan(dense_sum):
        assert np.isnan(sparse_sum), f"Expected NaN but got {sparse_sum}"
    else:
        assert sparse_sum == dense_sum

# Run the test with the specific failing input
print("Testing with [np.nan]...")
try:
    data = [np.nan]
    arr = SparseArray(data, fill_value=0.0)

    sparse_sum = arr.sum(skipna=False)
    dense_sum = arr.to_dense().sum()

    if np.isnan(dense_sum):
        assert np.isnan(sparse_sum), f"Expected NaN but got {sparse_sum}"
    else:
        assert sparse_sum == dense_sum
    print("Test passed with [np.nan]")
except AssertionError as e:
    print(f"Test failed with [np.nan]: {e}")

# Run more hypothesis tests
print("\nRunning hypothesis tests...")
from hypothesis import find
try:
    failing_example = find(
        st.lists(
            st.floats(allow_nan=True, allow_infinity=False) |
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
            min_size=1, max_size=100
        ),
        lambda data: test_sum_matches_dense_skipna_false(data) or True
    )
    print("No failures found")
except Exception as e:
    print(f"Found failure: {e}")