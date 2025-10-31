import numpy as np
import numpy.ma as ma

print("Testing the example from the documentation:")
print()

# Exact example from the docstring
a = [1, 2, 1000, 2, 3]
mask = [0, 0, 1, 0, 0]
masked_a = np.ma.masked_array(a, mask)

print(f"Input array a: {a}")
print(f"Mask: {mask}")
print(f"Masked array: {masked_a}")
print()

result = np.ma.unique(masked_a)
print(f"Result: {result}")
print(f"Result data: {result.data}")
print(f"Result mask: {ma.getmaskarray(result)}")
print(f"Number of masked values: {np.sum(ma.getmaskarray(result))}")
print()

# Expected from documentation:
print("Expected from documentation:")
print("  masked_array(data=[1, 2, 3, --],")
print("              mask=[False, False, False,  True],")
print("              fill_value=999999)")
print()

# The problem case
print("="*60)
print("Testing the bug report case again:")
print()

arr = np.array([32767, 32767, 32767], dtype=np.int16)
mask = np.array([True, False, True])
marr = ma.array(arr, mask=mask)

print(f"Array: {arr}")
print(f"Mask: {mask}")
print(f"Masked array: {marr}")
print()

unique_result = ma.unique(marr)
print(f"Result: {unique_result}")
print(f"Result mask: {ma.getmaskarray(unique_result)}")
print(f"Number of masked values: {np.sum(ma.getmaskarray(unique_result))}")

# Let's understand what's happening
print("\n" + "="*60)
print("Understanding the issue:")
print()

# What does np.unique do on this?
print("What np.unique returns on the masked array:")
np_result = np.unique(marr)
print(f"  np.unique(marr) = {np_result}")
print(f"  Type: {type(np_result)}")
if hasattr(np_result, 'mask'):
    print(f"  Mask: {np_result.mask}")
print()

# The issue is that when the underlying data has duplicates
# AND some of those duplicates are masked, np.unique returns
# multiple instances of that value, preserving masks
print("The bug occurs when:")
print("1. The underlying data has duplicate values (e.g., 32767 appears 3 times)")
print("2. Some instances are masked (positions 0 and 2) and some are not (position 1)")
print("3. np.unique() returns all unique combinations of (data, mask)")
print("4. So we get both masked and unmasked versions of 32767")