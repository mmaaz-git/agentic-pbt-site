#!/usr/bin/env python3
import numpy as np
import dask.array as da
from hypothesis import given, strategies as st, settings

# First, let's test the property-based test from the bug report
@given(
    st.lists(st.integers(min_value=-100, max_value=100), min_size=3, max_size=20),
    st.integers(min_value=-50, max_value=50)
)
@settings(max_examples=100)
def test_diff_append_should_use_dask_broadcast(arr_list, append_val):
    """Test that dask.diff with append produces correct results"""
    arr_np = np.array(arr_list)
    arr_da = da.from_array(arr_np, chunks=len(arr_list) // 2 + 1)

    result = da.diff(arr_da, append=append_val)
    dask_result = result.compute()
    numpy_result = np.diff(arr_np, append=append_val)

    np.testing.assert_array_equal(dask_result, numpy_result)
    print(f"✓ Test passed with arr_list={arr_list[:3]}... and append_val={append_val}")

# Run the property test
print("Running property-based test...")
test_diff_append_should_use_dask_broadcast()
print("Property test completed successfully!")

# Now test the reproducing code from the bug report
print("\n" + "="*50)
print("Testing the bug reproduction code...")
print("="*50)

arr = da.from_array(np.array([1, 2, 3, 4, 5]), chunks=3)
result_append = da.diff(arr, append=10)

print(f"Type of result_append: {type(result_append)}")
print(f"Result is a dask array: {isinstance(result_append, da.Array)}")
print(f"Computed result: {result_append.compute()}")

# Compare with numpy
numpy_arr = np.array([1, 2, 3, 4, 5])
numpy_result = np.diff(numpy_arr, append=10)
print(f"NumPy result: {numpy_result}")

# Test with prepend as well
print("\n" + "="*50)
print("Testing with both prepend and append...")
print("="*50)

result_both = da.diff(arr, prepend=0, append=10)
print(f"Type of result_both: {type(result_both)}")
print(f"Result is a dask array: {isinstance(result_both, da.Array)}")
print(f"Computed result: {result_both.compute()}")

numpy_result_both = np.diff(numpy_arr, prepend=0, append=10)
print(f"NumPy result: {numpy_result_both}")

# Let's also check if the function produces correct results
print("\n" + "="*50)
print("Checking correctness of results...")
print("="*50)

# Test multiple cases
test_cases = [
    ([1, 2, 3, 4, 5], 10, None),
    ([1, 2, 3, 4, 5], None, 0),
    ([1, 2, 3, 4, 5], 10, 0),
    ([10, 20, 30], 40, -10),
]

for arr_vals, append_val, prepend_val in test_cases:
    np_arr = np.array(arr_vals)
    da_arr = da.from_array(np_arr, chunks=2)

    da_result = da.diff(da_arr, append=append_val, prepend=prepend_val)
    np_result = np.diff(np_arr, append=append_val, prepend=prepend_val)

    da_computed = da_result.compute()

    try:
        np.testing.assert_array_equal(da_computed, np_result)
        print(f"✓ Test passed: arr={arr_vals}, append={append_val}, prepend={prepend_val}")
        print(f"  Result: {da_computed}")
    except AssertionError as e:
        print(f"✗ Test failed: arr={arr_vals}, append={append_val}, prepend={prepend_val}")
        print(f"  Dask result: {da_computed}")
        print(f"  NumPy result: {np_result}")