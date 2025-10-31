from pandas.arrays import SparseArray

# Create an empty SparseArray
arr = SparseArray([])

# Print information about the array
print(f"Array: {arr}")
print(f"Length: {len(arr)}")
print(f"Shape: {arr.shape}")
print(f"Dtype: {arr.dtype}")

# Try to access the density property - this should crash
try:
    density = arr.density
    print(f"Density: {density}")
except ZeroDivisionError as e:
    print(f"ZeroDivisionError: {e}")