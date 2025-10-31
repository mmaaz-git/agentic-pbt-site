#!/usr/bin/env python3
"""Test if numpy.strings.lower has the same issue"""

import numpy as np

# Test with the new numpy.strings API (if available)
try:
    import numpy.strings as npstr

    print("Testing with numpy.strings.lower (recommended API):")
    print("=" * 50)

    test_char = 'İ'
    arr = np.array([test_char], dtype=str)

    print(f"Input: {arr}")
    print(f"Input dtype: {arr.dtype}")

    result = npstr.lower(arr)
    print(f"Result: {result}")
    print(f"Result dtype: {result.dtype}")
    print(f"Result[0]: {result[0]!r}")

    expected = test_char.lower()
    print(f"Python lower: {expected!r}")
    print(f"Match: {result[0] == expected}")

except (ImportError, AttributeError) as e:
    print(f"numpy.strings not available or doesn't have lower: {e}")

# Also test if we pre-allocate a larger dtype
print("\n" + "=" * 50)
print("Testing with pre-allocated larger dtype:")

test_char = 'İ'
# Create array with larger dtype to accommodate expansion
arr = np.array([test_char], dtype='<U5')  # Allocate space for 5 characters
print(f"Input array with dtype <U5: {arr}")
print(f"Array dtype: {arr.dtype}")

import numpy.char as char
result = char.lower(arr)
print(f"Result: {result}")
print(f"Result dtype: {result.dtype}")
print(f"Result[0]: {result[0]!r}")
print(f"Python lower: {test_char.lower()!r}")
print(f"Match: {result[0] == test_char.lower()}")