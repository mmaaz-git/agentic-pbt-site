from hypothesis import given, strategies as st, settings, example
import numpy as np
from pandas.core.arrays.sparse import SparseArray

@given(st.lists(
    st.floats(allow_nan=True, allow_infinity=False),
    min_size=1, max_size=100
))
@example(data=[np.nan])  # The minimal failing case
@settings(max_examples=100)
def test_sum_matches_dense_skipna_false(data):
    """Test that SparseArray.sum(skipna=False) properly handles NaN values."""
    arr = SparseArray(data, fill_value=0.0)

    sparse_sum = arr.sum(skipna=False)
    dense_sum = arr.to_dense().sum()

    # Both should be NaN if there's any NaN in the data
    if np.isnan(dense_sum):
        assert np.isnan(sparse_sum), f"Expected NaN but got {sparse_sum} for data={data}"
    else:
        # Check if they're close (accounting for floating point precision)
        assert np.isclose(sparse_sum, dense_sum, rtol=1e-10, equal_nan=True), \
            f"Expected {dense_sum} but got {sparse_sum} for data={data}"

if __name__ == "__main__":
    # Run the test
    test_sum_matches_dense_skipna_false()