from hypothesis import given, strategies as st, settings
import pandas.core.arrays as arr
import numpy as np

@given(st.lists(st.integers(min_value=0, max_value=1000), min_size=0, max_size=100))
@settings(max_examples=500)
def test_sparsearray_empty_and_edge_cases(values):
    if len(values) == 0:
        sparse = arr.SparseArray([], fill_value=0)
        assert len(sparse) == 0
        try:
            density = sparse.density
            # If we get here, check if it's 0 or NaN
            assert density == 0 or np.isnan(density)
        except ZeroDivisionError:
            print("ZeroDivisionError occurred for empty array")
            raise
    else:
        sparse = arr.SparseArray(values, fill_value=0)
        assert len(sparse) == len(values)

# Run the test
test_sparsearray_empty_and_edge_cases()