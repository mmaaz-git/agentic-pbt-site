import numpy as np
import numpy.ma as ma

# Create a masked array with multiple masked values having different underlying data
arr = ma.array([999, 9223372036854775807, 888], mask=[True, False, True])
unique_result = ma.unique(arr)

print(f"Input array: {arr}")
print(f"Input data: {arr.data}")
print(f"Input mask: {arr.mask}")
print()
print(f"Unique result: {unique_result}")
print(f"Unique data: {unique_result.data}")
print(f"Unique mask: {unique_result.mask}")
print()

# Count masked values in the result
masked_count = sum(1 for val in unique_result if ma.is_masked(val))
print(f"Number of masked values in unique result: {masked_count}")
print(f"Expected: At most 1 (per documentation: 'Masked values are considered the same element')")
print(f"Actual: {masked_count}")
print()

if masked_count > 1:
    print("BUG CONFIRMED: Multiple masked values returned instead of treating all masked as one element")
else:
    print("No bug detected")