#!/usr/bin/env python3
"""Test the reported bug in pandas.core.strings.accessor.cat_core"""

import numpy as np
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')
import pandas.core.strings.accessor as accessor

print("=== Testing cat_core with null byte separator ===")

# Simple reproduction case
arr1 = np.array(['hello'], dtype=object)
arr2 = np.array(['world'], dtype=object)

result = accessor.cat_core([arr1, arr2], '\x00')

print(f"Input arrays: {arr1}, {arr2}")
print(f"Separator: {repr('\x00')}")
print(f"Result: {repr(result[0])}")
print(f"Expected: {repr('hello\x00world')}")
print(f"Match: {result[0] == 'hello\x00world'}")
print()

# Test with other special characters for comparison
print("=== Testing with other special characters ===")
test_cases = [
    ('\\x00', '\x00', 'null byte'),
    ('\\t', '\t', 'tab'),
    ('\\n', '\n', 'newline'),
    ('\\r', '\r', 'carriage return'),
    ('|', '|', 'pipe'),
    (',', ',', 'comma'),
    ('', '', 'empty string'),
]

for escape, sep, desc in test_cases:
    result = accessor.cat_core([arr1, arr2], sep)
    expected = f'hello{sep}world'
    match = result[0] == expected
    print(f"{desc:20} ({escape:5}): result={repr(result[0]):20} expected={repr(expected):20} match={match}")

print()
print("=== Testing with multiple arrays ===")
arr3 = np.array(['foo'], dtype=object)
result = accessor.cat_core([arr1, arr2, arr3], '\x00')
expected = 'hello\x00world\x00foo'
print(f"Arrays: {arr1}, {arr2}, {arr3}")
print(f"Result: {repr(result[0])}")
print(f"Expected: {repr(expected)}")
print(f"Match: {result[0] == expected}")

print()
print("=== Testing Hypothesis example ===")
# Test the specific failing case from hypothesis
arrays = [
    np.array(['s0_0'], dtype=object),
    np.array(['s1_0'], dtype=object),
]

result = accessor.cat_core(arrays, '\x00')
expected = 's0_0\x00s1_0'
print(f"Arrays: {arrays}")
print(f"Result: {repr(result[0])}")
print(f"Expected: {repr(expected)}")
print(f"Match: {result[0] == expected}")