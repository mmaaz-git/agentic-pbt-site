import numpy as np
import numpy.ma as ma

print("Testing the manual reproduction case from the bug report...")
print()

arr = np.array([32767, 32767, 32767], dtype=np.int16)
mask = np.array([True, False, True])
marr = ma.array(arr, mask=mask)

print(f"Input array: {arr}")
print(f"Input mask: {mask}")
print(f"Masked array: {marr}")
print()

unique_result = ma.unique(marr)
print(f'Result: {unique_result}')
print(f'Result data: {unique_result.data}')
print(f'Result mask: {ma.getmaskarray(unique_result)}')
print(f'Number of masked values: {np.sum(ma.getmaskarray(unique_result))}')
print()

print("Expected behavior according to documentation:")
print("  - All masked values should be treated as the same element")
print("  - Therefore, there should be at most 1 masked value in the result")
print()

print("Actual behavior:")
print(f"  - We got {np.sum(ma.getmaskarray(unique_result))} masked values in the result")

# Let's also test another case to understand the behavior better
print("\n" + "="*60)
print("Additional test case with different underlying values:")
arr2 = np.array([1, 2, 3, 4], dtype=np.int16)
mask2 = np.array([True, False, True, False])
marr2 = ma.array(arr2, mask=mask2)

print(f"Input array: {arr2}")
print(f"Input mask: {mask2}")
print(f"Masked array: {marr2}")

unique_result2 = ma.unique(marr2)
print(f'Result: {unique_result2}')
print(f'Result mask: {ma.getmaskarray(unique_result2)}')
print(f'Number of masked values: {np.sum(ma.getmaskarray(unique_result2))}')

print("\nAnalysis:")
print("It appears that ma.unique() is treating each masked value with")
print("a different underlying data value as distinct, rather than")
print("collapsing all masked values into a single masked element.")