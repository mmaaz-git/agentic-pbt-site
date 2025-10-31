from pandas.core.arrays.sparse import SparseArray
import numpy as np

# Test case 1: Single element array with fill_value=0
print("Test 1: Single element array [0]")
print("-" * 40)
try:
    arr = SparseArray([0])
    print(f"SparseArray([0]).argmin() = {arr.argmin()}")
except Exception as e:
    print(f"SparseArray([0]).argmin() raised: {type(e).__name__}: {e}")

dense = np.array([0])
print(f"numpy.array([0]).argmin() = {dense.argmin()}")
print()

# Test case 2: Multiple equal values with fill_value
print("Test 2: Array [5,5,5] with fill_value=5")
print("-" * 40)
try:
    arr = SparseArray([5, 5, 5], fill_value=5)
    print(f"SparseArray([5,5,5], fill_value=5).argmin() = {arr.argmin()}")
except Exception as e:
    print(f"SparseArray([5,5,5], fill_value=5).argmin() raised: {type(e).__name__}: {e}")

dense = np.array([5, 5, 5])
print(f"numpy.array([5,5,5]).argmin() = {dense.argmin()}")
print()

# Test case 3: argmax with same conditions
print("Test 3: argmax on [0,0,0] with fill_value=0")
print("-" * 40)
try:
    arr = SparseArray([0, 0, 0], fill_value=0)
    print(f"SparseArray([0,0,0], fill_value=0).argmax() = {arr.argmax()}")
except Exception as e:
    print(f"SparseArray([0,0,0], fill_value=0).argmax() raised: {type(e).__name__}: {e}")

dense = np.array([0, 0, 0])
print(f"numpy.array([0,0,0]).argmax() = {dense.argmax()}")
print()

# Test case 4: Working case - mixed values
print("Test 4: Working case - [0,1,0] with fill_value=0")
print("-" * 40)
arr = SparseArray([0, 1, 0], fill_value=0)
print(f"SparseArray([0,1,0], fill_value=0).argmin() = {arr.argmin()}")
print(f"SparseArray([0,1,0], fill_value=0).argmax() = {arr.argmax()}")