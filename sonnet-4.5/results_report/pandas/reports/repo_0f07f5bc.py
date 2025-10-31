import numpy as np
from pandas.core.arrays import SparseArray

# Create a SparseArray with values equal to the fill_value
sparse = SparseArray([0.0, 0.0, 0.0, 0.0, 0.0], fill_value=0.0)
print(f"Original sparse array: {sparse.to_dense()}")
print(f"Original fill_value: {sparse.fill_value}")
print(f"Original sparse values: {sparse.sp_values}")
print(f"Original sparse index: {sparse.sp_index}")
print()

# Create a new SparseArray from the first one with different fill_value
new_sparse = SparseArray(sparse, fill_value=1.0)
print(f"After changing fill_value to 1.0: {new_sparse.to_dense()}")
print(f"New fill_value: {new_sparse.fill_value}")
print(f"New sparse values: {new_sparse.sp_values}")
print(f"New sparse index: {new_sparse.sp_index}")
print()

# Check if the values are preserved
expected = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
actual = new_sparse.to_dense()
print(f"Expected: {expected}")
print(f"Actual: {actual}")
print(f"Are they equal? {np.allclose(expected, actual)}")

# This assertion will fail
assert np.allclose(new_sparse.to_dense(), [0.0, 0.0, 0.0, 0.0, 0.0]), \
    f"Expected {[0.0, 0.0, 0.0, 0.0, 0.0]} but got {new_sparse.to_dense()}"