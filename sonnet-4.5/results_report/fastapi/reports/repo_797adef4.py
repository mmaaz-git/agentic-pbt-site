import numpy as np
import dask.array as da

# Create a simple 2D array
arr = np.array([[1, 2, 3], [4, 5, 6]])
darr = da.from_array(arr, chunks=2)

print("Original array:")
print(arr)
print("\n" + "="*60 + "\n")

# Test with prepend parameter
print("Testing prepend parameter:")
result_prepend = da.diff(darr, prepend=0, axis=1)
print(f"Type of prepend broadcast result: {type(result_prepend._meta)}")
print(f"Result with prepend=0:")
print(result_prepend.compute())
print("\n" + "="*60 + "\n")

# Test with append parameter
print("Testing append parameter:")
result_append = da.diff(darr, append=0, axis=1)
print(f"Type of append broadcast result: {type(result_append._meta)}")
print(f"Result with append=0:")
print(result_append.compute())
print("\n" + "="*60 + "\n")

# Show the problem more directly
print("Direct inspection of the issue:")
print(f"prepend uses: dask.array.broadcast_to (from dask.array.core)")
print(f"append uses:  np.broadcast_to (from numpy)")

# Demonstrate the inconsistency in lazy evaluation
print("\n" + "="*60 + "\n")
print("Demonstrating the lazy evaluation inconsistency:")

# Create a large array to show potential memory issues
large_arr = np.random.rand(100, 100)
large_darr = da.from_array(large_arr, chunks=(10, 10))

# With prepend (dask broadcast_to - maintains lazy evaluation)
result_prepend_large = da.diff(large_darr, prepend=0, axis=1)
print(f"With prepend: Result is dask array: {isinstance(result_prepend_large, da.Array)}")

# With append (np.broadcast_to - breaks lazy evaluation for append part)
result_append_large = da.diff(large_darr, append=0, axis=1)
print(f"With append: Result is dask array: {isinstance(result_append_large, da.Array)}")

# But the internal broadcast operation differs
print("\nThis inconsistency breaks the dask lazy evaluation model for the append parameter.")