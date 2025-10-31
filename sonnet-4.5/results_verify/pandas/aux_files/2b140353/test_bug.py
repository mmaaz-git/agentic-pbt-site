#!/usr/bin/env python3
"""Test the reported SparseArray.cumsum bug"""

# First, let's test the simple reproduction case
print("Testing simple reproduction case:")
try:
    from pandas.arrays import SparseArray
    import numpy as np

    sparse = SparseArray([1, 2, 3])
    print(f"Created SparseArray: {sparse}")
    print(f"Fill value: {sparse.fill_value}")
    print(f"_null_fill_value: {sparse._null_fill_value}")

    print("\nAttempting to call cumsum()...")
    result = sparse.cumsum()
    print(f"Result: {result}")
except RecursionError as e:
    print(f"RecursionError caught: {str(e)[:100]}...")
except Exception as e:
    print(f"Other error: {type(e).__name__}: {e}")

print("\n" + "="*50 + "\n")

# Now test with the hypothesis test
print("Testing with hypothesis property-based test:")
try:
    from hypothesis import given, strategies as st
    import numpy as np
    from pandas.arrays import SparseArray

    @given(st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=100))
    def test_cumsum_does_not_crash(values):
        arr = np.array(values)
        sparse = SparseArray(arr)

        cumsum_result = sparse.cumsum()
        assert len(cumsum_result) == len(sparse)
        return True

    # Try running the test with a specific example
    print("Testing with [1]:")
    test_result = test_cumsum_does_not_crash([1])
    print(f"Test passed: {test_result}")

except RecursionError as e:
    print(f"RecursionError caught: {str(e)[:100]}...")
except Exception as e:
    print(f"Other error: {type(e).__name__}: {e}")

print("\n" + "="*50 + "\n")

# Let's also test with different fill values
print("Testing with different fill values:")

# Test with NaN fill value (should work according to bug report)
try:
    import pandas as pd
    sparse_nan = SparseArray([1.0, 2.0, 3.0], fill_value=np.nan)
    print(f"SparseArray with NaN fill_value: {sparse_nan}")
    print(f"Fill value: {sparse_nan.fill_value}")
    print(f"_null_fill_value: {sparse_nan._null_fill_value}")
    result_nan = sparse_nan.cumsum()
    print(f"cumsum() result: {result_nan}")
    print("SUCCESS: NaN fill_value works")
except Exception as e:
    print(f"FAILED with NaN fill_value: {type(e).__name__}: {e}")

print()

# Test with 0 fill value (should fail according to bug report)
try:
    sparse_zero = SparseArray([1, 2, 3], fill_value=0)
    print(f"SparseArray with 0 fill_value: {sparse_zero}")
    print(f"Fill value: {sparse_zero.fill_value}")
    print(f"_null_fill_value: {sparse_zero._null_fill_value}")
    result_zero = sparse_zero.cumsum()
    print(f"cumsum() result: {result_zero}")
    print("SUCCESS: 0 fill_value works")
except RecursionError as e:
    print(f"FAILED with 0 fill_value: RecursionError - {str(e)[:50]}...")
except Exception as e:
    print(f"FAILED with 0 fill_value: {type(e).__name__}: {e}")