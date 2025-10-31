from pandas.arrays import SparseArray
import numpy as np
import sys

print("Test 1: Integer SparseArray with fill_value=0 (explicit)")
try:
    arr = SparseArray([1, 2, 3], fill_value=0)
    print(f"Created array: {arr}")
    print(f"Fill value: {arr.fill_value}")
    result = arr.cumsum()
    print(f"Cumsum result: {result}")
except RecursionError as e:
    print(f"RecursionError occurred: {e}")
except Exception as e:
    print(f"Other error: {type(e).__name__}: {e}")

print("\nTest 2: Integer SparseArray (default fill_value=0)")
try:
    arr_int = SparseArray([1, 2, 3])
    print(f"Created array: {arr_int}")
    print(f"Fill value: {arr_int.fill_value}")
    result = arr_int.cumsum()
    print(f"Cumsum result: {result}")
except RecursionError as e:
    print(f"RecursionError occurred: {e}")
except Exception as e:
    print(f"Other error: {type(e).__name__}: {e}")

print("\nTest 3: Float SparseArray (default fill_value=nan)")
try:
    arr_float = SparseArray([1.0, 2.0, 3.0])
    print(f"Created array: {arr_float}")
    print(f"Fill value: {arr_float.fill_value}")
    result = arr_float.cumsum()
    print(f"Cumsum result: {result}")
    print(f"Success! Expected result: [1.0, 3.0, 6.0]")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nTest 4: Testing with a single element [0]")
try:
    arr = SparseArray([0])
    print(f"Created array: {arr}")
    print(f"Fill value: {arr.fill_value}")
    result = arr.cumsum()
    print(f"Cumsum result: {result}")
except RecursionError as e:
    print(f"RecursionError occurred: {e}")
except Exception as e:
    print(f"Other error: {type(e).__name__}: {e}")

print("\nTest 5: Regular numpy array cumsum for comparison")
np_arr = np.array([1, 2, 3])
print(f"Numpy array: {np_arr}")
print(f"Numpy cumsum: {np.cumsum(np_arr)}")