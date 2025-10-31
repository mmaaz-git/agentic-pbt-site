from pandas.arrays import SparseArray

# Test the exact reproduction case
arr = SparseArray([])
print(f"Array: {arr}")
print(f"Length: {len(arr)}")

try:
    density = arr.density
    print(f"Density: {density}")
except ZeroDivisionError as e:
    print(f"ZeroDivisionError: {e}")