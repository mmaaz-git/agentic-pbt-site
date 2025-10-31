import numpy as np
import numpy.char as char

# Test how other numpy.char operations handle expansion

# Test 1: char.add (concatenation) - should expand
arr1 = np.array(['test'], dtype='U4')
arr2 = np.array(['ing'], dtype='U3')
result = char.add(arr1, arr2)
print(f"char.add result: {result[0]!r}, dtype: {result.dtype}")
print(f"Expected: 'testing', length: 7")

# Test 2: char.multiply - should expand
arr = np.array(['ab'], dtype='U2')
result = char.multiply(arr, 3)
print(f"\nchar.multiply result: {result[0]!r}, dtype: {result.dtype}")
print(f"Expected: 'ababab', length: 6")

# Test 3: char.upper with expansion (testing if issue is specific to title)
# Using the same ligature character
arr = np.array(['ﬁ test'], dtype='U6')
upper_result = char.upper(arr)
title_result = char.title(arr)
print(f"\nInput: {arr[0]!r}, dtype: {arr.dtype}")
print(f"char.upper result: {upper_result[0]!r}, dtype: {upper_result.dtype}")
print(f"char.title result: {title_result[0]!r}, dtype: {title_result.dtype}")
print(f"Python upper: {'ﬁ test'.upper()!r}")
print(f"Python title: {'ﬁ test'.title()!r}")