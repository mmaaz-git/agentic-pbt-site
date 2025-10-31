import numpy as np
import numpy.ma as ma

# Create the failing test case
arr = np.array([32767, 32767, 32767], dtype=np.int16)
mask = np.array([True, False, True])
marr = ma.array(arr, mask=mask)

print("Input array:", arr)
print("Input mask:", mask)
print("Masked array:", marr)
print()

# Call ma.unique
unique_result = ma.unique(marr)
print("Result from ma.unique():", unique_result)
print("Result data:", unique_result.data)
print("Result mask:", ma.getmaskarray(unique_result))
print("Number of masked values in result:", np.sum(ma.getmaskarray(unique_result)))
print()

# According to documentation, masked values should be considered the same element
# Expected: At most 1 masked value in the result
# Actual: Multiple masked values are returned
print("Expected: At most 1 masked value (per documentation)")
print("Actual:", np.sum(ma.getmaskarray(unique_result)), "masked values")
print()
print("BUG: ma.unique() returns multiple masked values when it should treat")
print("all masked values as the same element and return at most one masked value.")