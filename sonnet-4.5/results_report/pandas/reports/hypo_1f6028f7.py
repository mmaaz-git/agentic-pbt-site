from hypothesis import given, strategies as st
import numpy as np
from pandas.arrays import SparseArray

@given(st.lists(
    st.one_of(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10),
        st.just(np.nan)
    ),
    min_size=1,
    max_size=50
))
def test_max_min_skipna_false(values):
    arr = SparseArray(values, fill_value=0.0)

    if any(np.isnan(v) for v in values):
        # Test max
        sparse_max = arr.max(skipna=False)
        dense_max = np.max(arr.to_dense())

        assert np.isnan(sparse_max) and np.isnan(dense_max), \
            f"When array contains NaN and skipna=False, max should return NaN. Got {sparse_max} for values {values}"

        # Test min
        sparse_min = arr.min(skipna=False)
        dense_min = np.min(arr.to_dense())

        assert np.isnan(sparse_min) and np.isnan(dense_min), \
            f"When array contains NaN and skipna=False, min should return NaN. Got {sparse_min} for values {values}"

# Run the test
if __name__ == "__main__":
    test_max_min_skipna_false()