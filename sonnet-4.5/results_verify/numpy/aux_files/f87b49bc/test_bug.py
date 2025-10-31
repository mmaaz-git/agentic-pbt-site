import numpy as np
import numpy.ma as ma

# Test the basic reproduction case
print("=== Basic Reproduction Case ===")
arr = ma.array([999, 9223372036854775807, 888], mask=[True, False, True])
unique_result = ma.unique(arr)

print(f"Input: {arr}")
print(f"Input data: {arr.data}")
print(f"Input mask: {arr.mask}")
print(f"Unique result: {unique_result}")
print(f"Unique data: {unique_result.data}")
print(f"Unique mask: {unique_result.mask}")

masked_count = sum(1 for val in unique_result if ma.is_masked(val))
print(f"Number of masked values in result: {masked_count}")
print(f"Expected: At most 1")
print(f"Actual: {masked_count}")

# Test with the example from the documentation
print("\n=== Documentation Example ===")
a = [1, 2, 1000, 2, 3]
mask = [0, 0, 1, 0, 0]
masked_a = ma.masked_array(a, mask)
print(f"Input: {masked_a}")
unique_doc = ma.unique(masked_a)
print(f"Unique: {unique_doc}")
masked_count_doc = sum(1 for val in unique_doc if ma.is_masked(val))
print(f"Number of masked values: {masked_count_doc}")

# Test with multiple masked values having the same underlying data
print("\n=== Same Underlying Data Test ===")
arr2 = ma.array([999, 100, 999], mask=[True, False, True])
unique_result2 = ma.unique(arr2)
print(f"Input: {arr2}")
print(f"Input data: {arr2.data}")
print(f"Input mask: {arr2.mask}")
print(f"Unique result: {unique_result2}")
masked_count2 = sum(1 for val in unique_result2 if ma.is_masked(val))
print(f"Number of masked values: {masked_count2}")

# Another test with different underlying values
print("\n=== Different Underlying Values Test ===")
arr3 = ma.array([1, 2, 3, 4, 5], mask=[True, False, True, False, True])
unique_result3 = ma.unique(arr3)
print(f"Input: {arr3}")
print(f"Input data: {arr3.data}")
print(f"Input mask: {arr3.mask}")
print(f"Unique result: {unique_result3}")
print(f"Unique data: {unique_result3.data}")
print(f"Unique mask: {unique_result3.mask}")
masked_count3 = sum(1 for val in unique_result3 if ma.is_masked(val))
print(f"Number of masked values: {masked_count3}")
