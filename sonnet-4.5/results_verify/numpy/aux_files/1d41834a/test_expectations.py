import numpy as np

# Test what happens with regular numpy arrays and copy behavior
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
non_contiguous = arr[::2, ::2]

print("=== Testing regular np.array behavior ===")
print(f"Non-contiguous input: {non_contiguous.flags.contiguous}")

# Test np.array with copy=False
arr_nocopy = np.array(non_contiguous, copy=False)
print(f"np.array(non_contiguous, copy=False) contiguous: {arr_nocopy.flags.contiguous}")

# Test np.array with copy=True
arr_copy = np.array(non_contiguous, copy=True)
print(f"np.array(non_contiguous, copy=True) contiguous: {arr_copy.flags.contiguous}")

# Test np.ascontiguousarray
arr_contig = np.ascontiguousarray(non_contiguous)
print(f"np.ascontiguousarray(non_contiguous) contiguous: {arr_contig.flags.contiguous}")

# Check if the arrays share memory
print(f"\nMemory sharing:")
print(f"non_contiguous shares memory with arr_nocopy: {np.shares_memory(non_contiguous, arr_nocopy)}")
print(f"non_contiguous shares memory with arr_copy: {np.shares_memory(non_contiguous, arr_copy)}")