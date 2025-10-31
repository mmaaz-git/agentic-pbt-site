#!/usr/bin/env python3
"""Test script to reproduce the numpy.strings.find null character bug"""

import numpy as np
import numpy.strings as nps

print("Testing numpy.strings.find with null character '\\x00':")
print("=" * 60)

test_cases = ['', 'abc', 'a\x00b', '\x00', 'hello\x00world', '\x00start', 'end\x00']

for s in test_cases:
    arr = np.array([s], dtype=str)
    np_find = nps.find(arr, '\x00')[0]
    py_find = s.find('\x00')
    match = "✓" if np_find == py_find else "✗"
    print(f"find({repr(s):20}, '\\x00'): Python={py_find:3}, NumPy={np_find:3} {match}")

print("\n" + "=" * 60)
print("\nTesting with other substrings for comparison:")
print("=" * 60)

# Test with regular characters to see if the issue is specific to null character
test_strings = ['abc', 'hello', 'test']
search_chars = ['x', 'z', 'q']  # Characters not in the test strings

for s in test_strings:
    for char in search_chars:
        arr = np.array([s], dtype=str)
        np_find = nps.find(arr, char)[0]
        py_find = s.find(char)
        match = "✓" if np_find == py_find else "✗"
        print(f"find({repr(s):10}, {repr(char)}): Python={py_find:3}, NumPy={np_find:3} {match}")

print("\n" + "=" * 60)
print("\nTesting the hypothesis test case:")
print("=" * 60)

# The specific failing case from the bug report
arr = np.array(['abc'], dtype=str)
sub = '\x00'
np_result = nps.find(arr, sub)[0]
py_result = 'abc'.find(sub)
print(f"Bug report case: arr=['abc'], sub='\\x00'")
print(f"  Python result: {py_result}")
print(f"  NumPy result:  {np_result}")
print(f"  Match: {'✓' if np_result == py_result else '✗ MISMATCH'}")

# Also test with bounds
print("\n" + "=" * 60)
print("\nTesting with start/end bounds:")
print("=" * 60)

test_string = 'abcdefg'
arr = np.array([test_string], dtype=str)

for start in [0, 2, 5]:
    for end in [None, 3, 7]:
        np_find = nps.find(arr, '\x00', start, end)[0]
        py_find = test_string.find('\x00', start, end)
        match = "✓" if np_find == py_find else "✗"
        print(f"find('abcdefg', '\\x00', {start}, {end}): Python={py_find:3}, NumPy={np_find:3} {match}")