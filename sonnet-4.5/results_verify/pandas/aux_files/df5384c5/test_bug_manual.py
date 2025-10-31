import numpy as np
from pandas.arrays import SparseArray

# Manual reproduction
print("Manual reproduction of the bug:")
print("="*50)

arr = SparseArray([0, 1, 2, 2], fill_value=2)

print(f"Array: {arr}")
print(f"Array data type: {type(arr)}")
print(f"Fill value: {arr.fill_value}")
print(f"to_dense(): {arr.to_dense()}")
print(f"to_dense() type: {type(arr.to_dense())}")
print(f"Expected nonzero positions (to_dense().nonzero()[0]): {arr.to_dense().nonzero()[0]}")
print(f"Actual nonzero positions (arr.nonzero()[0]): {arr.nonzero()[0]}")

try:
    assert np.array_equal(arr.nonzero()[0], arr.to_dense().nonzero()[0])
    print("\nAssertion passed - arrays are equal")
except AssertionError:
    print("\nAssertion failed - arrays are NOT equal")
    print(f"  Expected: {arr.to_dense().nonzero()[0]}")
    print(f"  Got:      {arr.nonzero()[0]}")

# Let's also test some other cases
print("\n" + "="*50)
print("Testing additional cases:")
print("="*50)

# Case 1: fill_value=0 (standard sparse array)
arr1 = SparseArray([0, 1, 2, 0], fill_value=0)
print(f"\nCase 1: {arr1}, fill_value={arr1.fill_value}")
print(f"  to_dense().nonzero()[0]: {arr1.to_dense().nonzero()[0]}")
print(f"  arr.nonzero()[0]:        {arr1.nonzero()[0]}")
print(f"  Match: {np.array_equal(arr1.nonzero()[0], arr1.to_dense().nonzero()[0])}")

# Case 2: fill_value=1 (nonzero)
arr2 = SparseArray([0, 1, 2, 1], fill_value=1)
print(f"\nCase 2: {arr2}, fill_value={arr2.fill_value}")
print(f"  to_dense().nonzero()[0]: {arr2.to_dense().nonzero()[0]}")
print(f"  arr.nonzero()[0]:        {arr2.nonzero()[0]}")
print(f"  Match: {np.array_equal(arr2.nonzero()[0], arr2.to_dense().nonzero()[0])}")

# Case 3: All elements equal to fill_value
arr3 = SparseArray([3, 3, 3, 3], fill_value=3)
print(f"\nCase 3: {arr3}, fill_value={arr3.fill_value}")
print(f"  to_dense().nonzero()[0]: {arr3.to_dense().nonzero()[0]}")
print(f"  arr.nonzero()[0]:        {arr3.nonzero()[0]}")
print(f"  Match: {np.array_equal(arr3.nonzero()[0], arr3.to_dense().nonzero()[0])}")