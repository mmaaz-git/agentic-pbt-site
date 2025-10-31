import numpy as np
import numpy.ma as ma

print("=== Understanding np.unique vs ma.unique ===")

# Test case with different underlying values
data = [999, 9223372036854775807, 888]
mask = [True, False, True]
arr = ma.array(data, mask=mask)

print(f"Original array: {arr}")
print(f"Original data: {arr.data}")
print(f"Original mask: {arr.mask}")

# What does np.unique do with the raw data?
print("\n--- np.unique on raw data ---")
raw_unique = np.unique(arr.data)
print(f"np.unique on data: {raw_unique}")

# What does ma.unique do?
print("\n--- ma.unique on masked array ---")
ma_unique = ma.unique(arr)
print(f"ma.unique result: {ma_unique}")
print(f"ma.unique data: {ma_unique.data}")
print(f"ma.unique mask: {ma_unique.mask}")

# Test with return_index
print("\n--- With return_index ---")
unique_vals, indices = ma.unique(arr, return_index=True)
print(f"Unique values: {unique_vals}")
print(f"Indices: {indices}")
print(f"Original values at indices: {[arr[i] for i in indices]}")

# Test with return_inverse
print("\n--- With return_inverse ---")
unique_vals, inverse = ma.unique(arr, return_inverse=True)
print(f"Unique values: {unique_vals}")
print(f"Inverse indices: {inverse}")
print(f"Reconstruction: {unique_vals[inverse]}")

# Test what happens when masked values have the same underlying data
print("\n=== Test with same underlying data for masked values ===")
arr2 = ma.array([999, 100, 999], mask=[True, False, True])
print(f"Array with same underlying masked data: {arr2}")
unique2 = ma.unique(arr2)
print(f"Unique result: {unique2}")
masked_count = sum(1 for val in unique2 if ma.is_masked(val))
print(f"Number of masked values: {masked_count}")
