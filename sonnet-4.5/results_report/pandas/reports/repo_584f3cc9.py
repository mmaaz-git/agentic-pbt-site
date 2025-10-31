from pandas.core.arrays.sparse import SparseArray

# Create a SparseArray with non-null fill value (0)
sparse_arr = SparseArray([1, 2, 3], fill_value=0)
print(f"Sparse array: {sparse_arr.to_dense()}")
print(f"Fill value: {sparse_arr.fill_value}")
print(f"_null_fill_value: {sparse_arr._null_fill_value}")

# Attempt to calculate cumulative sum
try:
    result = sparse_arr.cumsum()
    print(f"Success: {result.to_dense()}")
except RecursionError as e:
    print(f"RecursionError: maximum recursion depth exceeded")
    print(f"Error type: {type(e).__name__}")