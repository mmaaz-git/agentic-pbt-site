import pandas as pd
import numpy as np

print("Demonstrating ArrowExtensionArray.take([]) bug\n")
print("=" * 60)

# Create an ArrowExtensionArray with integer data
print("Creating ArrowExtensionArray with int64[pyarrow] dtype:")
arr = pd.array([1, 2, 3], dtype='int64[pyarrow]')
print(f"arr = pd.array([1, 2, 3], dtype='int64[pyarrow]')")
print(f"arr = {arr}")
print(f"arr.dtype = {arr.dtype}")
print()

# Try to take with empty indices - this should crash
print("Attempting to take with empty indices:")
print("result = arr.take([])")
print()

try:
    result = arr.take([])
    print(f"Result: {result}")
    print(f"Result length: {len(result)}")
    print(f"Result dtype: {result.dtype}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    print()

print("=" * 60)
print("\nComparison with regular NumPy-backed array (which works correctly):\n")

# Show that regular arrays work fine
print("Creating regular int64 array:")
regular_arr = pd.array([1, 2, 3], dtype='int64')
print(f"regular_arr = pd.array([1, 2, 3], dtype='int64')")
print(f"regular_arr = {regular_arr}")
print(f"regular_arr.dtype = {regular_arr.dtype}")
print()

print("Attempting to take with empty indices:")
print("result = regular_arr.take([])")
try:
    result = regular_arr.take([])
    print(f"Result: {result}")
    print(f"Result length: {len(result)}")
    print(f"Result dtype: {result.dtype}")
    print("\nSUCCESS: Regular array handles empty indices correctly")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

print()
print("=" * 60)
print("\nDiagnosis of the issue:")
print("When indices=[], np.asanyarray([]) creates a float64 array by default:")
indices = []
indices_array = np.asanyarray(indices)
print(f"indices = {indices}")
print(f"np.asanyarray(indices) = {indices_array}")
print(f"np.asanyarray(indices).dtype = {indices_array.dtype}")
print()
print("PyArrow cannot handle float64 indices with integer arrays,")
print("causing the 'no kernel matching input types' error.")