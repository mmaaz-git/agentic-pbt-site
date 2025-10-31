#!/usr/bin/env python3
"""Deep dive into numpy.sum behavior with null bytes"""

import numpy as np

print("Testing numpy.sum string concatenation behavior")
print("=" * 60)

# Test 1: Direct numpy.sum with strings containing null bytes
print("\nTest 1: Direct string concatenation with null bytes")
arr = np.array(['hello\x00', 'world'], dtype=object)
result = np.sum(arr)
print(f"np.sum(['hello\\x00', 'world']) = {repr(result)}")
print(f"Expected: 'hello\\x00world'")
print(f"Actual length: {len(result)}, Expected length: {len('hello\x00world')}")

# Test 2: Mixed array with scalar separator
print("\nTest 2: Mixed array (arrays + scalar string)")
arr1 = np.array(['a', 'b'], dtype=object)
mixed = np.array([arr1, '\x00', arr1], dtype=object)
result = np.sum(mixed, axis=0)
print(f"Array structure: [array, '\\x00', array]")
print(f"Result: {repr(result)}")
print(f"Expected: ['a\\x00a', 'b\\x00b']")

# Test 3: All strings with null byte
print("\nTest 3: All plain strings")
arr = np.array(['hello', '\x00', 'world'], dtype=object)
result = np.sum(arr)
print(f"np.sum(['hello', '\\x00', 'world']) = {repr(result)}")
print(f"Expected: 'hello\\x00world'")

# Test 4: Test if the issue is specific to mixed types
print("\nTest 4: Mixed types investigation")

# Create the exact structure used in cat_core
list_of_columns = [
    np.array(['hello'], dtype=object),
    np.array(['world'], dtype=object)
]
sep = '\x00'

# cat_core method (fails)
list_with_sep = [sep] * (2 * len(list_of_columns) - 1)
list_with_sep[::2] = list_of_columns
print(f"\nlist_with_sep structure: {list_with_sep}")
print(f"Types in list: {[type(x) for x in list_with_sep]}")

arr_with_sep = np.asarray(list_with_sep, dtype=object)
print(f"Array shape: {arr_with_sep.shape}")
print(f"Array content: {arr_with_sep}")

result = np.sum(arr_with_sep, axis=0)
print(f"Result after sum: {repr(result)}")
print(f"Expected: ['hello\\x00world']")

# Test 5: Alternative approach with all arrays
print("\nTest 5: All arrays approach")
sep_array = np.array([sep], dtype=object)
list_with_sep2 = [sep_array] * 3
list_with_sep2[::2] = list_of_columns
result2 = np.sum(np.asarray(list_with_sep2, dtype=object), axis=0)
print(f"Result with array separator: {repr(result2)}")

# Test 6: Character by character analysis
print("\nTest 6: Character analysis of concatenation")
str1 = "hello"
str2 = "\x00"
str3 = "world"

# Manual concatenation
manual = str1 + str2 + str3
print(f"Manual concat: {repr(manual)}, len={len(manual)}")

# NumPy sum on simple array
simple_arr = np.array([str1, str2, str3], dtype=object)
numpy_sum = np.sum(simple_arr)
print(f"NumPy sum of strings: {repr(numpy_sum)}, len={len(numpy_sum)}")

# Check each character
print("\nByte-by-byte comparison:")
print(f"Manual: {[hex(ord(c)) for c in manual]}")
print(f"NumPy:  {[hex(ord(c)) for c in numpy_sum]}")