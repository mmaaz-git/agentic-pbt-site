#!/usr/bin/env python3
"""Test to reproduce the reported bug with dask.array.ravel() on empty arrays."""

import dask.array as da
import numpy as np
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays, array_shapes
import traceback


# First, let's test the specific failing case mentioned in the bug report
print("=" * 60)
print("Testing specific failing case: np.empty((3, 0), dtype=np.float64)")
print("=" * 60)

try:
    arr = np.empty((3, 0), dtype=np.float64)
    print(f"NumPy array shape: {arr.shape}")
    print(f"NumPy array size: {arr.size}")

    # Test NumPy's ravel behavior
    np_raveled = np.ravel(arr)
    print(f"NumPy ravel result shape: {np_raveled.shape}")
    print(f"NumPy ravel result size: {np_raveled.size}")

    # Test Dask's ravel behavior
    darr = da.from_array(arr, chunks=2)
    print(f"\nDask array shape: {darr.shape}")
    print(f"Dask array chunks: {darr.chunks}")

    print("\nAttempting to ravel Dask array...")
    raveled = da.ravel(darr)
    print(f"Dask ravel created (lazy evaluation)")

    # This should trigger the error
    result = raveled.compute()
    print(f"Dask ravel result shape: {result.shape}")
    print(f"Dask ravel result size: {result.size}")

except Exception as e:
    print(f"\nERROR encountered:")
    print(f"Exception type: {type(e).__name__}")
    print(f"Exception message: {str(e)}")
    print("\nTraceback:")
    traceback.print_exc()


# Now let's run the property-based test
print("\n" + "=" * 60)
print("Running property-based test")
print("=" * 60)

failures = []

@given(
    arr=arrays(
        dtype=np.float64,
        shape=array_shapes(min_dims=1, max_dims=2, min_side=0, max_side=10)
    )
)
@settings(max_examples=100, deadline=None)
def test_ravel_preserves_elements(arr):
    try:
        darr = da.from_array(arr, chunks=2)
        raveled = da.ravel(darr)
        assert raveled.ndim == 1
        assert raveled.size == arr.size
        result = raveled.compute()
        expected = np.ravel(arr)
        np.testing.assert_array_equal(result, expected)
    except Exception as e:
        failures.append((arr.shape, str(e)))
        raise

try:
    test_ravel_preserves_elements()
    print("Property-based test completed successfully!")
except Exception as e:
    print(f"Property-based test failed!")
    print(f"Number of failures encountered: {len(failures)}")
    if failures:
        print("\nFailed shapes:")
        for shape, error in failures[:5]:  # Show first 5 failures
            print(f"  Shape {shape}: {error[:100]}...")


# Let's test a few more edge cases with empty arrays
print("\n" + "=" * 60)
print("Testing additional empty array edge cases")
print("=" * 60)

test_cases = [
    (0,),        # 1D empty array
    (0, 0),      # 2D fully empty array
    (3, 0),      # 2D with one zero dimension (reported case)
    (0, 3),      # 2D with other zero dimension
    (5, 0, 2),   # 3D with zero in middle
]

for shape in test_cases:
    print(f"\nTesting shape {shape}:")
    try:
        arr = np.empty(shape, dtype=np.float64)
        np_result = np.ravel(arr)
        print(f"  NumPy ravel: {arr.shape} -> {np_result.shape}")

        darr = da.from_array(arr, chunks=2)
        da_raveled = da.ravel(darr)
        da_result = da_raveled.compute()
        print(f"  Dask ravel: {darr.shape} -> {da_result.shape}")

        assert np.array_equal(da_result, np_result), "Results don't match!"
        print(f"  ✓ Success")

    except Exception as e:
        print(f"  ✗ Failed: {type(e).__name__}: {str(e)[:100]}")