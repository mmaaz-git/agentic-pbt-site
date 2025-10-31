import numpy as np
import pandas as pd
from pandas.core.arrays.sparse import SparseArray
from hypothesis import given, strategies as st, settings
import traceback

print("Testing pandas SparseArray argmin/argmax bug")
print("=" * 60)

# First, let's verify we have the right pandas version
print(f"Pandas version: {pd.__version__}")
print()

# Test 1: Basic reproduction case from the bug report
print("Test 1: Basic case with [0] and fill_value=0")
try:
    arr = SparseArray([0], fill_value=0)
    dense = arr.to_dense()

    print(f"SparseArray: {arr}")
    print(f"Dense array: {dense}")
    print(f"npoints (number of sparse values): {arr.npoints}")

    print("Calling arr.argmin()...")
    result = arr.argmin()
    print(f"argmin() returned: {result}")
except Exception as e:
    print(f"ERROR: {e}")
    print(f"Error type: {type(e).__name__}")
    traceback.print_exc()

print()
print("-" * 60)

# Test 2: Another case from the bug report
print("Test 2: Array with all same values [5, 5, 5, 5] and fill_value=5")
try:
    arr = SparseArray([5, 5, 5, 5], fill_value=5)
    dense = arr.to_dense()

    print(f"SparseArray: {arr}")
    print(f"Dense array: {dense}")
    print(f"npoints (number of sparse values): {arr.npoints}")

    print("Calling arr.argmin()...")
    result_min = arr.argmin()
    print(f"argmin() returned: {result_min}")

    print("Calling arr.argmax()...")
    result_max = arr.argmax()
    print(f"argmax() returned: {result_max}")
except Exception as e:
    print(f"ERROR: {e}")
    print(f"Error type: {type(e).__name__}")
    traceback.print_exc()

print()
print("-" * 60)

# Test 3: Compare with NumPy behavior
print("Test 3: Compare with NumPy behavior for array of same values")
np_arr = np.array([5, 5, 5, 5])
print(f"NumPy array: {np_arr}")
print(f"NumPy argmin(): {np_arr.argmin()}")
print(f"NumPy argmax(): {np_arr.argmax()}")

print()
print("-" * 60)

# Test 4: Property-based test from the bug report
print("Test 4: Running property-based test")
@given(
    st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=50),
    st.integers(min_value=-100, max_value=100)
)
@settings(max_examples=300)
def test_argmin_argmax_consistency(data, fill_value):
    arr = SparseArray(data, fill_value=fill_value)
    dense = arr.to_dense()

    if len(arr) > 0:
        # The assertion from the bug report
        assert arr[arr.argmin()] == dense[dense.argmin()]
        assert arr[arr.argmax()] == dense[dense.argmax()]

try:
    test_argmin_argmax_consistency()
    print("Property test completed successfully")
except Exception as e:
    print(f"Property test failed: {e}")
    traceback.print_exc()

print()
print("-" * 60)

# Test 5: Check case where not all values are fill_value (should work)
print("Test 5: Normal case with mixed values [1, 2, 3, 4] and fill_value=0")
try:
    arr = SparseArray([1, 2, 3, 4], fill_value=0)
    dense = arr.to_dense()

    print(f"SparseArray: {arr}")
    print(f"Dense array: {dense}")
    print(f"npoints (number of sparse values): {arr.npoints}")

    print(f"argmin() returned: {arr.argmin()}")
    print(f"argmax() returned: {arr.argmax()}")
    print(f"dense.argmin() returned: {dense.argmin()}")
    print(f"dense.argmax() returned: {dense.argmax()}")
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()