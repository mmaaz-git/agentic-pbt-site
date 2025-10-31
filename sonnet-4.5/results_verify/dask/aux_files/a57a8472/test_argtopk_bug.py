#!/usr/bin/env python3
"""Test script to reproduce the argtopk bug"""

import numpy as np
import dask.array as da
from hypothesis import given, strategies as st, settings
from hypothesis.extra import numpy as hnp


# First, let's reproduce the specific failing case mentioned
def test_specific_failing_case():
    print("Testing specific failing case: arr=da.from_array([1,2,3,4,5], chunks=2), k=5")
    try:
        arr = da.from_array(np.array([1, 2, 3, 4, 5], dtype=np.int32), chunks=2)
        print(f"Array: {arr}")
        print(f"Array shape: {arr.shape}")
        print(f"Array chunks: {arr.chunks}")
        result = da.argtopk(arr, 5).compute()
        print(f"Result: {result}")
        print(f"Result length: {len(result)}")
        print("SUCCESS: No error occurred")
    except ValueError as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    return True


# Test with single chunk (should work according to bug report)
def test_single_chunk():
    print("\nTesting with single chunk (should work):")
    try:
        arr = da.from_array(np.array([1, 2, 3, 4, 5], dtype=np.int32), chunks=5)
        print(f"Array chunks: {arr.chunks}")
        result = da.argtopk(arr, 5).compute()
        print(f"Result: {result}")
        print("SUCCESS: Single chunk works")
        return True
    except Exception as e:
        print(f"ERROR with single chunk: {e}")
        return False


# Test with k < array size (should work)
def test_k_less_than_size():
    print("\nTesting with k < array size:")
    try:
        arr = da.from_array(np.array([1, 2, 3, 4, 5], dtype=np.int32), chunks=2)
        result = da.argtopk(arr, 3).compute()
        print(f"Result for k=3: {result}")
        print("SUCCESS: k < array size works")
        return True
    except Exception as e:
        print(f"ERROR with k < size: {e}")
        return False


# Now let's run the property-based test
@st.composite
def dask_array_for_argtopk(draw):
    shape = draw(st.tuples(st.integers(min_value=5, max_value=30)))
    dtype = draw(st.sampled_from([np.int32, np.float64]))

    if dtype == np.float64:
        np_arr = draw(hnp.arrays(
            dtype, shape,
            elements=st.floats(min_value=-1000, max_value=1000,
                             allow_nan=False, allow_infinity=False)
        ))
    else:
        np_arr = draw(hnp.arrays(
            dtype, shape,
            elements=st.integers(min_value=-1000, max_value=1000)
        ))

    chunks = draw(st.integers(min_value=2, max_value=max(3, shape[0] // 2)))
    k = draw(st.integers(min_value=1, max_value=min(10, shape[0])))

    return da.from_array(np_arr, chunks=chunks), k


@given(dask_array_for_argtopk())
@settings(max_examples=50)  # Reduced for faster testing
def test_argtopk_returns_correct_size(data):
    arr, k = data
    try:
        result = da.argtopk(arr, k).compute()
        assert len(result) == k
    except ValueError as e:
        if "too many values to unpack" in str(e):
            print(f"\nFound bug case: shape={arr.shape}, chunks={arr.chunks}, k={k}")
            raise


if __name__ == "__main__":
    print("=" * 70)
    print("Testing dask.array.argtopk bug")
    print("=" * 70)

    # Run specific tests
    test_specific_failing_case()
    test_single_chunk()
    test_k_less_than_size()

    # Run property-based test
    print("\nRunning property-based test (max 50 examples):")
    try:
        test_argtopk_returns_correct_size()
        print("Property-based test completed without finding errors")
    except Exception as e:
        print(f"Property-based test found error: {e}")