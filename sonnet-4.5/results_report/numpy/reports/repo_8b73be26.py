from pandas.arrays import SparseArray

# Create a sparse array with fill_value=0 (non-null value)
arr = SparseArray([1, 0, 2], fill_value=0)
print(f"Original array: {arr}")
print(f"Array values: {arr.to_dense()}")
print(f"Fill value: {arr.fill_value}")

# Try to compute cumulative sum - this should cause infinite recursion
try:
    result = arr.cumsum()
    print(f"Cumulative sum result: {result}")
    print(f"Result values: {result.to_dense()}")
except RecursionError as e:
    print(f"RecursionError occurred: {e}")