#!/usr/bin/env python3
"""Test script to reproduce the periodic/reflect bug with large depth"""

import numpy as np
import dask.array as da
from dask.array.overlap import periodic, reflect
import traceback

print("Testing periodic and reflect functions with depth larger than array size")
print("=" * 70)

# Test 1: Reproduce the exact bug from the report
print("\n1. Testing periodic with arr_size=5, axis=0, depth=6:")
try:
    arr = da.arange(25).reshape(5, 5).rechunk(3)
    print(f"   Array shape: {arr.shape}, chunks: {arr.chunks}")
    result = periodic(arr, axis=0, depth=6)
    print(f"   Result shape: {result.shape}")
    print("   SUCCESS: No error occurred!")
except ValueError as e:
    print(f"   ERROR: {e}")
except Exception as e:
    print(f"   UNEXPECTED ERROR: {e}")
    traceback.print_exc()

# Test 2: Test with depth equal to array size
print("\n2. Testing periodic with depth equal to array size (depth=5):")
try:
    arr = da.arange(25).reshape(5, 5).rechunk(3)
    result = periodic(arr, axis=0, depth=5)
    print(f"   Result shape: {result.shape}")
    print("   SUCCESS: No error occurred!")
except ValueError as e:
    print(f"   ERROR: {e}")
except Exception as e:
    print(f"   UNEXPECTED ERROR: {e}")

# Test 3: Test with depth less than array size
print("\n3. Testing periodic with depth less than array size (depth=3):")
try:
    arr = da.arange(25).reshape(5, 5).rechunk(3)
    result = periodic(arr, axis=0, depth=3)
    print(f"   Result shape: {result.shape}")
    print("   SUCCESS: No error occurred!")
except ValueError as e:
    print(f"   ERROR: {e}")
except Exception as e:
    print(f"   UNEXPECTED ERROR: {e}")

# Test 4: Test reflect function with large depth
print("\n4. Testing reflect with depth larger than array size (depth=6):")
try:
    arr = da.arange(25).reshape(5, 5).rechunk(3)
    result = reflect(arr, axis=0, depth=6)
    print(f"   Result shape: {result.shape}")
    print("   SUCCESS: No error occurred!")
except ValueError as e:
    print(f"   ERROR: {e}")
except Exception as e:
    print(f"   UNEXPECTED ERROR: {e}")

# Test 5: Compare with NumPy's pad behavior
print("\n5. Compare with NumPy's pad function:")
try:
    np_arr = np.arange(5)
    print(f"   Original NumPy array: {np_arr}")
    result = np.pad(np_arr, (6, 6), mode='wrap')  # 'wrap' is equivalent to periodic
    print(f"   np.pad with depth=6, mode='wrap': shape={result.shape}, result={result}")
    print("   NumPy handles depth > array_size without error")
except Exception as e:
    print(f"   NumPy ERROR: {e}")

# Test 6: Test the hypothesis test case
print("\n6. Running the Hypothesis test case from the bug report:")
from hypothesis import given, strategies as st, settings

@given(
    arr_size=st.integers(min_value=5, max_value=30),
    axis=st.integers(min_value=0, max_value=1),
    depth=st.integers(min_value=1, max_value=10),
    chunk_size=st.integers(min_value=3, max_value=10)
)
@settings(max_examples=10)  # Reduced for quick testing
def test_periodic_size_increase(arr_size, axis, depth, chunk_size):
    shape = (arr_size, arr_size)
    arr = da.arange(np.prod(shape)).reshape(shape).rechunk(chunk_size)

    try:
        result = periodic(arr, axis, depth)
        expected_shape = list(shape)
        expected_shape[axis] += 2 * depth
        assert result.shape == tuple(expected_shape), f"Shape mismatch: {result.shape} != {expected_shape}"
    except ValueError:
        return  # Hypothesis test should return None
    except AssertionError:
        raise  # Re-raise assertion errors

# Run a few test cases
print("   Running property-based tests...")
failures = []
errors = []
for _ in range(5):
    try:
        test_periodic_size_increase()
    except AssertionError as e:
        failures.append(str(e))
    except Exception as e:
        errors.append(str(e))

if failures:
    print(f"   Found {len(failures)} failures")
    for f in failures[:2]:  # Show first 2 failures
        print(f"     - {f}")
elif errors:
    print(f"   Found {len(errors)} errors (likely from ValueError)")
else:
    print("   All test cases passed")

# Test specifically the reported failing case
print("\n7. Testing the specific failing case from the report:")
arr_size, axis, depth, chunk_size = 5, 0, 6, 3
print(f"   arr_size={arr_size}, axis={axis}, depth={depth}, chunk_size={chunk_size}")
shape = (arr_size, arr_size)
arr = da.arange(np.prod(shape)).reshape(shape).rechunk(chunk_size)
try:
    result = periodic(arr, axis, depth)
    expected_shape = list(shape)
    expected_shape[axis] += 2 * depth
    print(f"   Expected shape: {expected_shape}, Got: {result.shape}")
except ValueError as e:
    print(f"   Confirmed ValueError: {e}")