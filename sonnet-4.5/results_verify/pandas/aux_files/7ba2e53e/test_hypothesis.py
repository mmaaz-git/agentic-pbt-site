#!/usr/bin/env python3

from hypothesis import given, strategies as st, settings, example
import numpy as np
from pandas.core.util.hashing import hash_array, combine_hash_arrays

@given(
    st.lists(st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=100), min_size=1, max_size=10),
)
@settings(max_examples=100)  # Reduced for faster testing
@example([[0], [0, 0]])  # The specific failing case
def test_combine_hash_arrays_assertion(array_lists):
    arrays = [np.array(arr, dtype=np.int64) for arr in array_lists]
    hash_arrays = [hash_array(arr) for arr in arrays]
    num_items = len(hash_arrays)
    try:
        result = combine_hash_arrays(iter(hash_arrays), num_items)
        assert result.dtype == np.uint64
        print(f"✓ Passed for input with lengths: {[len(arr) for arr in array_lists]}")
    except ValueError as e:
        print(f"✗ Failed for input with lengths: {[len(arr) for arr in array_lists]}")
        print(f"  Arrays: {array_lists}")
        print(f"  Error: {e}")
        raise

if __name__ == "__main__":
    print("Running hypothesis test...")
    test_combine_hash_arrays_assertion()