import numpy as np
from pandas.arrays import SparseArray

arr = SparseArray([0, 1, 2, 2], fill_value=2)

print(f"Array: {arr}")
print(f"Fill value: {arr.fill_value}")
print(f"to_dense(): {arr.to_dense()}")
print(f"Expected nonzero positions (from to_dense().nonzero()): {arr.to_dense().nonzero()[0]}")
print(f"Actual nonzero positions (from arr.nonzero()): {arr.nonzero()[0]}")

try:
    assert np.array_equal(arr.nonzero()[0], arr.to_dense().nonzero()[0])
    print("Test passed: sparse.nonzero() matches to_dense().nonzero()")
except AssertionError:
    print("AssertionError: sparse.nonzero() does not match to_dense().nonzero()")