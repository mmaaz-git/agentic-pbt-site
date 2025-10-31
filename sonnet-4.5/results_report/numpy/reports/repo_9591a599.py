#!/usr/bin/env python3
"""Minimal reproduction of null byte handling bug in numpy.strings"""

import numpy as np
import numpy.strings as ns

# Create array with a single null byte character
arr = np.array(['\x00'], dtype=np.str_)

print("Testing numpy.strings functions with null byte ('\\x00'):")
print("=" * 60)

# Test case conversion functions
print(f"upper:")
print(f"  NumPy result: {repr(ns.upper(arr)[0])}")
print(f"  Python result: {repr('\x00'.upper())}")
print(f"  Match: {ns.upper(arr)[0] == '\x00'.upper()}")
print()

print(f"lower:")
print(f"  NumPy result: {repr(ns.lower(arr)[0])}")
print(f"  Python result: {repr('\x00'.lower())}")
print(f"  Match: {ns.lower(arr)[0] == '\x00'.lower()}")
print()

print(f"capitalize:")
print(f"  NumPy result: {repr(ns.capitalize(arr)[0])}")
print(f"  Python result: {repr('\x00'.capitalize())}")
print(f"  Match: {ns.capitalize(arr)[0] == '\x00'.capitalize()}")
print()

print(f"title:")
print(f"  NumPy result: {repr(ns.title(arr)[0])}")
print(f"  Python result: {repr('\x00'.title())}")
print(f"  Match: {ns.title(arr)[0] == '\x00'.title()}")
print()

print(f"swapcase:")
print(f"  NumPy result: {repr(ns.swapcase(arr)[0])}")
print(f"  Python result: {repr('\x00'.swapcase())}")
print(f"  Match: {ns.swapcase(arr)[0] == '\x00'.swapcase()}")
print()

# Test stripping functions
print(f"strip:")
print(f"  NumPy result: {repr(ns.strip(arr)[0])}")
print(f"  Python result: {repr('\x00'.strip())}")
print(f"  Match: {ns.strip(arr)[0] == '\x00'.strip()}")
print()

# Test partition function
left, mid, right = ns.partition(arr, 'X')
python_result = '\x00'.partition('X')
print(f"partition (with separator 'X'):")
print(f"  NumPy result: ({repr(left[0])}, {repr(mid[0])}, {repr(right[0])})")
print(f"  Python result: {repr(python_result)}")
print(f"  Match: {(left[0], mid[0], right[0]) == python_result}")
print()

# Test with null byte in the middle of string (should work correctly)
print("=" * 60)
print("Testing with null byte in middle of string ('hel\\x00lo'):")
arr2 = np.array(['hel\x00lo'], dtype=np.str_)
print(f"upper: NumPy={repr(ns.upper(arr2)[0])}, Python={repr('hel\x00lo'.upper())}")
print(f"Match: {ns.upper(arr2)[0] == 'hel\x00lo'.upper()}")