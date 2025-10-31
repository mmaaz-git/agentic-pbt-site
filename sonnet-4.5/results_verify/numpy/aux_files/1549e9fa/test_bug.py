#!/usr/bin/env python3
"""Test script to reproduce the reported bug"""

import numpy as np
import pandas.core.strings.accessor as accessor

# Test 1: Basic reproduction test from bug report
print("=" * 60)
print("Test 1: Basic bug reproduction")
print("=" * 60)

arr1 = np.array(['hello'], dtype=object)
arr2 = np.array(['world'], dtype=object)

result = accessor.cat_core([arr1, arr2], '\x00')

print(f"Input arrays: {arr1}, {arr2}")
print(f"Separator: {repr('\x00')} (null byte)")
print(f"Result: {repr(result[0])}")
print(f"Expected: {repr('hello\x00world')}")
print(f"Bug present: {result[0] != 'hello\x00world'}")

# Test 2: Compare with other separators
print("\n" + "=" * 60)
print("Test 2: Compare different separators")
print("=" * 60)

separators = [
    ('|', 'pipe'),
    ('\t', 'tab'),
    ('\n', 'newline'),
    (' ', 'space'),
    ('\x00', 'null byte'),
    ('\x01', 'SOH character'),
    ('', 'empty string')
]

for sep, name in separators:
    result = accessor.cat_core([arr1, arr2], sep)
    expected = 'hello' + sep + 'world'
    matches = result[0] == expected
    print(f"{name:15} | sep={repr(sep):6} | result={repr(result[0]):20} | matches={matches}")

# Test 3: Property-based test from bug report
print("\n" + "=" * 60)
print("Test 3: Property-based test (minimal case)")
print("=" * 60)

from hypothesis import given, strategies as st, settings

@given(
    array_length=st.integers(min_value=1, max_value=10),
    num_arrays=st.integers(min_value=2, max_value=4),
)
@settings(max_examples=50)
def test_cat_core_preserves_separator(array_length, num_arrays):
    sep = '\x00'

    arrays = [
        np.array([f's{i}_{j}' for j in range(array_length)], dtype=object)
        for i in range(num_arrays)
    ]

    result = accessor.cat_core(arrays, sep)

    for i in range(array_length):
        expected = sep.join([arrays[j][i] for j in range(num_arrays)])
        assert result[i] == expected, \
            f"At index {i}: expected {repr(expected)}, got {repr(result[i])}"

try:
    test_cat_core_preserves_separator()
    print("Property-based test PASSED")
except AssertionError as e:
    print(f"Property-based test FAILED: {e}")

# Test 4: Direct numpy.sum behavior
print("\n" + "=" * 60)
print("Test 4: Direct numpy.sum behavior investigation")
print("=" * 60)

# Test how numpy.sum handles different separators
test_arrays = [
    np.array(['a', 'b'], dtype=object),
    np.array(['c', 'd'], dtype=object)
]

for sep, name in [('|', 'pipe'), ('\x00', 'null byte')]:
    # Method 1: Scalar separator (as used in cat_core)
    list_with_sep = [sep] * 3
    list_with_sep[::2] = test_arrays
    arr_with_sep = np.asarray(list_with_sep, dtype=object)
    result1 = np.sum(arr_with_sep, axis=0)

    # Method 2: Array separator
    sep_array = np.full(2, sep, dtype=object)
    list_with_sep2 = [sep_array] * 3
    list_with_sep2[::2] = test_arrays
    arr_with_sep2 = np.asarray(list_with_sep2, dtype=object)
    result2 = np.sum(arr_with_sep2, axis=0)

    print(f"\n{name} separator:")
    print(f"  Method 1 (scalar sep): {repr(result1)}")
    print(f"  Method 2 (array sep):  {repr(result2)}")
    print(f"  Expected: {repr(np.array(['a' + sep + 'c', 'b' + sep + 'd'], dtype=object))}")

# Test 5: More edge cases
print("\n" + "=" * 60)
print("Test 5: Additional edge cases")
print("=" * 60)

# Multiple null bytes
result = accessor.cat_core([arr1, arr2], '\x00\x00')
print(f"Double null byte sep: {repr(result[0])}, expected: {repr('hello\x00\x00world')}")

# Null byte in middle of separator
result = accessor.cat_core([arr1, arr2], 'X\x00Y')
print(f"'X\\x00Y' separator: {repr(result[0])}, expected: {repr('helloX\x00Yworld')}")

# Empty arrays
empty1 = np.array([''], dtype=object)
empty2 = np.array([''], dtype=object)
result = accessor.cat_core([empty1, empty2], '\x00')
print(f"Empty strings with null: {repr(result[0])}, expected: {repr('\x00')}")