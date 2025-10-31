#!/usr/bin/env python3
"""Test the hypothesis test from the bug report"""

from hypothesis import given, strategies as st, settings
import numpy as np
from dask.dataframe.dask_expr._expr import _sort_mixed

@given(st.lists(st.one_of(
    st.integers(),
    st.text(),
    st.tuples(st.integers()),
    st.just(None)
), min_size=1, max_size=50))
@settings(max_examples=100)
def test_sort_mixed_order(values):
    arr = np.array(values, dtype=object)
    result = _sort_mixed(arr)
    assert len(result) == len(arr)
    print(f"Tested: {values[:3]}... (len={len(values)})")

if __name__ == "__main__":
    print("Running hypothesis test...")
    try:
        test_sort_mixed_order()
        print("All hypothesis tests passed!")
    except Exception as e:
        print(f"Hypothesis test failed: {e}")