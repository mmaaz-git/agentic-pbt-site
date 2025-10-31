#!/usr/bin/env python3
"""
Property-based test that discovered the SparseArray.density ZeroDivisionError bug.
"""

from hypothesis import given, strategies as st, settings
import pandas.core.arrays as arr
import numpy as np

@given(st.lists(st.integers(min_value=0, max_value=1000), min_size=0, max_size=100))
@settings(max_examples=500)
def test_sparsearray_empty_and_edge_cases(values):
    if len(values) == 0:
        sparse = arr.SparseArray([], fill_value=0)
        assert len(sparse) == 0
        assert sparse.density == 0 or np.isnan(sparse.density)
    else:
        sparse = arr.SparseArray(values, fill_value=0)
        assert len(sparse) == len(values)

if __name__ == "__main__":
    # Run the test
    test_sparsearray_empty_and_edge_cases()