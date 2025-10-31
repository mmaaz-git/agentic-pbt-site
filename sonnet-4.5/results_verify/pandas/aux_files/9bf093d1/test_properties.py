import pandas.core.arrays as arr

# Test with empty array
sparse_empty = arr.SparseArray([], fill_value=0)
print("Empty array:")
print(f"  sp_index.npoints: {sparse_empty.sp_index.npoints}")
print(f"  sp_index.length: {sparse_empty.sp_index.length}")
print(f"  len(sparse): {len(sparse_empty)}")

# Test with non-empty array
sparse_nonempty = arr.SparseArray([0, 0, 1, 1, 1], fill_value=0)
print("\nNon-empty array [0, 0, 1, 1, 1]:")
print(f"  sp_index.npoints: {sparse_nonempty.sp_index.npoints}")
print(f"  sp_index.length: {sparse_nonempty.sp_index.length}")
print(f"  density: {sparse_nonempty.density}")

# Test with all sparse
sparse_all_sparse = arr.SparseArray([0, 0, 0], fill_value=0)
print("\nAll sparse [0, 0, 0]:")
print(f"  sp_index.npoints: {sparse_all_sparse.sp_index.npoints}")
print(f"  sp_index.length: {sparse_all_sparse.sp_index.length}")
print(f"  density: {sparse_all_sparse.density}")