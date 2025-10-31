import numpy as np
import numpy.strings as nps

# Test with explicit null character
s1 = 'a'
s2 = 'a\x00'

print(f"Python strings: s1={repr(s1)}, s2={repr(s2)}")
print(f"Python comparison: s1 < s2 = {s1 < s2}")
print(f"Python comparison: s1 != s2 = {s1 != s2}")
print(f"Python comparison: s1 >= s2 = {s1 >= s2}")
print()

# Create numpy arrays with different dtypes
arr1 = np.array([s1], dtype='U1')  # Fixed width 1
arr2 = np.array([s2], dtype='U2')  # Fixed width 2

print(f"Arrays with dtype U1 and U2:")
print(f"arr1 = {repr(arr1)}, dtype={arr1.dtype}")
print(f"arr2 = {repr(arr2)}, dtype={arr2.dtype}")
print(f"arr1[0] = {repr(arr1[0])}, len={len(str(arr1[0]))}")
print(f"arr2[0] = {repr(arr2[0])}, len={len(str(arr2[0]))}")
print()

# Test with strings module functions
print("numpy.strings comparisons:")
print(f"not_equal: {nps.not_equal(arr1, arr2)[0]}")
print(f"less: {nps.less(arr1, arr2)[0]}")
print(f"greater_equal: {nps.greater_equal(arr1, arr2)[0]}")
print()

# Try with same dtype but explicit null
arr1_same = np.array(['a'], dtype='U2')
arr2_same = np.array(['a\x00'], dtype='U2')
print(f"Arrays with same dtype U2:")
print(f"arr1_same = {repr(arr1_same)}, dtype={arr1_same.dtype}")
print(f"arr2_same = {repr(arr2_same)}, dtype={arr2_same.dtype}")
print(f"arr1_same[0] = {repr(arr1_same[0])}")
print(f"arr2_same[0] = {repr(arr2_same[0])}")
print()

print("numpy.strings comparisons (same dtype):")
print(f"not_equal: {nps.not_equal(arr1_same, arr2_same)[0]}")
print(f"less: {nps.less(arr1_same, arr2_same)[0]}")
print(f"greater_equal: {nps.greater_equal(arr1_same, arr2_same)[0]}")
print()

# Check if the null is preserved
print(f"Check null preservation in arr2_same[0]: bytes = {arr2_same[0].encode('utf-8')}")
print(f"Length of arr2_same[0]: {len(str(arr2_same[0]))}")