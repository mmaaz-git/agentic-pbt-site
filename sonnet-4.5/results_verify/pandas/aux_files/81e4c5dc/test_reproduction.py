import pandas as pd
import pyarrow as pa
import traceback

print("Testing ArrowExtensionArray.take() with empty indices")
print("=" * 60)

# Test 1: Empty array with empty indices
print("\nTest 1: Empty array with empty indices")
print("-" * 40)
try:
    arr = pd.array(pa.array([], type=pa.int64()), dtype=pd.ArrowDtype(pa.int64()))
    print(f"Created empty array: {arr}")
    print(f"Array type: {type(arr)}")
    print(f"Array dtype: {arr.dtype}")

    result = arr.take([])
    print(f"Result of arr.take([]): {result}")
    print(f"Result length: {len(result)}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
    traceback.print_exc()

# Test 2: Non-empty array with empty indices
print("\nTest 2: Non-empty array with empty indices")
print("-" * 40)
try:
    arr = pd.array(pa.array([1, 2, 3], type=pa.int64()), dtype=pd.ArrowDtype(pa.int64()))
    print(f"Created array: {arr}")
    print(f"Array type: {type(arr)}")
    print(f"Array dtype: {arr.dtype}")

    result = arr.take([])
    print(f"Result of arr.take([]): {result}")
    print(f"Result length: {len(result)}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
    traceback.print_exc()

# Test 3: Regular take operation (should work)
print("\nTest 3: Regular take operation (control test)")
print("-" * 40)
try:
    arr = pd.array(pa.array([1, 2, 3], type=pa.int64()), dtype=pd.ArrowDtype(pa.int64()))
    print(f"Created array: {arr}")

    result = arr.take([0, 2])
    print(f"Result of arr.take([0, 2]): {result}")
    print("Regular take operation works correctly")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
    traceback.print_exc()

# Test 4: Investigate numpy's behavior with empty arrays
print("\nTest 4: numpy.asanyarray() behavior with empty list")
print("-" * 40)
import numpy as np
empty_list = []
numpy_array = np.asanyarray(empty_list)
print(f"np.asanyarray([]) = {numpy_array}")
print(f"dtype: {numpy_array.dtype}")
print(f"shape: {numpy_array.shape}")