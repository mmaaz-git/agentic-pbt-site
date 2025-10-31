#!/usr/bin/env python3
"""Run the hypothesis test from the bug report"""

from hypothesis import given, strategies as st, settings
from pandas.arrays import SparseArray
import numpy as np

@given(st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=50))
@settings(max_examples=10)  # Limit examples since we know it will fail
def test_cumsum_consistent_with_dense(data):
    sparse = SparseArray(data)
    dense = sparse.to_dense()

    try:
        sparse_cumsum = sparse.cumsum()
        dense_cumsum = np.cumsum(dense)
        assert np.array_equal(sparse_cumsum.to_dense(), dense_cumsum)
        print(f"✓ Test passed for: {data[:5]}...")
    except RecursionError as e:
        print(f"✗ RecursionError for: {data[:5]}...")
        raise
    except Exception as e:
        print(f"✗ Other error for {data[:5]}...: {e}")
        raise

# Run the test
print("Running Hypothesis test...")
try:
    test_cumsum_consistent_with_dense()
    print("All tests passed!")
except Exception as e:
    print(f"Test failed: {e}")

# Also test the specific failing input mentioned
print("\nTesting specific failing input [1]:")
try:
    sparse = SparseArray([1])
    dense = sparse.to_dense()
    sparse_cumsum = sparse.cumsum()
    dense_cumsum = np.cumsum(dense)
    print(f"Sparse cumsum: {sparse_cumsum}")
    print(f"Dense cumsum: {dense_cumsum}")
    assert np.array_equal(sparse_cumsum.to_dense(), dense_cumsum)
except RecursionError as e:
    print(f"RecursionError: {e}")