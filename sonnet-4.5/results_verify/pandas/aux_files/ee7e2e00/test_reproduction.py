from pandas.core.arrays.sparse import SparseArray
import numpy as np

print("Testing SparseArray with all fill_values:")
try:
    arr = SparseArray([0])
    print(f"Sparse argmin: {arr.argmin()}")
except ValueError as e:
    print(f"Sparse argmin raised ValueError: {e}")

try:
    arr = SparseArray([0])
    print(f"Sparse argmax: {arr.argmax()}")
except ValueError as e:
    print(f"Sparse argmax raised ValueError: {e}")

print("\nComparing with numpy:")
dense = np.array([0])
print(f"Dense argmin: {dense.argmin()}")
print(f"Dense argmax: {dense.argmax()}")

print("\nTesting with multiple equal values:")
try:
    arr2 = SparseArray([5, 5, 5], fill_value=5)
    print(f"Sparse argmin for [5,5,5]: {arr2.argmin()}")
except ValueError as e:
    print(f"Sparse argmin for [5,5,5] raised ValueError: {e}")

dense2 = np.array([5, 5, 5])
print(f"Dense argmin for [5,5,5]: {dense2.argmin()}")

print("\nTesting with mixed values:")
arr3 = SparseArray([0, 1, 0], fill_value=0)
print(f"Sparse argmin for [0,1,0]: {arr3.argmin()}")
print(f"Sparse argmax for [0,1,0]: {arr3.argmax()}")

dense3 = np.array([0, 1, 0])
print(f"Dense argmin for [0,1,0]: {dense3.argmin()}")
print(f"Dense argmax for [0,1,0]: {dense3.argmax()}")