import pandas.core.arrays.sparse as sparse
import numpy as np

# Test case 1: Array [1] with fill_value=1
print("Test case 1: Array [1] with fill_value=1")
arr1 = sparse.SparseArray([1], fill_value=1)
print(f"SparseArray: {arr1}")
print(f"SparseArray.nonzero(): {arr1.nonzero()}")
print(f"Dense array.nonzero(): {arr1.to_dense().nonzero()}")
print()

# Test case 2: Array [1, 2, 3] with fill_value=2
print("Test case 2: Array [1, 2, 3] with fill_value=2")
arr2 = sparse.SparseArray([1, 2, 3], fill_value=2)
print(f"SparseArray: {arr2}")
print(f"SparseArray.nonzero(): {arr2.nonzero()}")
print(f"Dense array.nonzero(): {arr2.to_dense().nonzero()}")
print()

# Test case 3: More complex case to show the issue clearly
print("Test case 3: Array [0, 1, 2, 0, 2, 2, 0] with fill_value=2")
arr3 = sparse.SparseArray([0, 1, 2, 0, 2, 2, 0], fill_value=2)
print(f"SparseArray: {arr3}")
print(f"SparseArray.nonzero(): {arr3.nonzero()}")
print(f"Dense array.nonzero(): {arr3.to_dense().nonzero()}")