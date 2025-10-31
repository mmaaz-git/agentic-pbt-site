import numpy as np
from pandas.core.arrays.sparse import SparseArray

# Case 1: Single element array where element equals fill_value
print("Case 1: SparseArray([0], fill_value=0)")
arr = SparseArray([0], fill_value=0)
dense = arr.to_dense()

print(f"SparseArray: {arr}")
print(f"Dense array: {dense}")
print(f"npoints (sparse values): {arr.npoints}")
print(f"Length: {len(arr)}")

try:
    result = arr.argmin()
    print(f"argmin() result: {result}")
except Exception as e:
    print(f"argmin() raised: {type(e).__name__}: {e}")

try:
    result = arr.argmax()
    print(f"argmax() result: {result}")
except Exception as e:
    print(f"argmax() raised: {type(e).__name__}: {e}")

print("\nFor comparison, NumPy on the same data:")
np_arr = np.array([0])
print(f"NumPy array: {np_arr}")
print(f"np.argmin(): {np_arr.argmin()}")
print(f"np.argmax(): {np_arr.argmax()}")

print("\n" + "="*50 + "\n")

# Case 2: Multiple elements, all equal to fill_value
print("Case 2: SparseArray([5, 5, 5, 5], fill_value=5)")
arr2 = SparseArray([5, 5, 5, 5], fill_value=5)
dense2 = arr2.to_dense()

print(f"SparseArray: {arr2}")
print(f"Dense array: {dense2}")
print(f"npoints (sparse values): {arr2.npoints}")
print(f"Length: {len(arr2)}")

try:
    result = arr2.argmin()
    print(f"argmin() result: {result}")
except Exception as e:
    print(f"argmin() raised: {type(e).__name__}: {e}")

try:
    result = arr2.argmax()
    print(f"argmax() result: {result}")
except Exception as e:
    print(f"argmax() raised: {type(e).__name__}: {e}")

print("\nFor comparison, NumPy on the same data:")
np_arr2 = np.array([5, 5, 5, 5])
print(f"NumPy array: {np_arr2}")
print(f"np.argmin(): {np_arr2.argmin()}")
print(f"np.argmax(): {np_arr2.argmax()}")