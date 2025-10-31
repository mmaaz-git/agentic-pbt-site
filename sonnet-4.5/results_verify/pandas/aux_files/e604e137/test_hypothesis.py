#!/usr/bin/env python3
"""Run the hypothesis test from the bug report."""

from hypothesis import given, strategies as st, settings
import numpy as np
from pandas.arrays import SparseArray

@given(
    data=st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=50)
)
@settings(max_examples=100, print_blob=True)
def test_fill_value_setter_preserves_data(data):
    sparse = SparseArray(data, fill_value=0)
    original_dense = sparse.to_dense()

    sparse.fill_value = 999

    assert np.array_equal(sparse.to_dense(), original_dense), f"Data changed from {original_dense} to {sparse.to_dense()}"

# Run the test
if __name__ == "__main__":
    try:
        test_fill_value_setter_preserves_data()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")