import pandas.arrays as pa
import numpy as np

# Test what happens when creating SparseArray from dense array
dense = np.array([1, 0, 2, 0, 3])
print(f"Dense array: {dense}")

# Create SparseArray from dense without specifying fill_value
sparse_no_fill = pa.SparseArray(dense)
print(f"\nSparseArray from dense (no fill_value specified):")
print(f"  Array: {sparse_no_fill}")
print(f"  Fill value: {sparse_no_fill.fill_value}")
print(f"  _null_fill_value: {sparse_no_fill._null_fill_value}")

# Create SparseArray with fill_value=0
sparse_with_zero = pa.SparseArray([1, 0, 2, 0, 3], fill_value=0)
print(f"\nSparseArray with fill_value=0:")
print(f"  Array: {sparse_with_zero}")
print(f"  Fill value: {sparse_with_zero.fill_value}")
print(f"  _null_fill_value: {sparse_with_zero._null_fill_value}")

# Convert to dense and back
dense_from_sparse = sparse_with_zero.to_dense()
print(f"\nDense from sparse: {dense_from_sparse}")

# Create new SparseArray from dense
new_sparse = pa.SparseArray(dense_from_sparse)
print(f"New SparseArray from dense:")
print(f"  Array: {new_sparse}")
print(f"  Fill value: {new_sparse.fill_value}")
print(f"  _null_fill_value: {new_sparse._null_fill_value}")