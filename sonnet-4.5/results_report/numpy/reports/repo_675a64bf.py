#!/usr/bin/env python3
"""
Minimal reproduction case demonstrating numpy.char.title truncation bug
with Unicode ligatures that expand during case conversion.
"""

import numpy as np
import numpy.char as char

# Test case with the ligature 'ﬁ' (U+FB01) which title-cases to 'Fi' (2 chars)
arr = np.array(['ﬁ test'], dtype=str)
result = char.title(arr)

print(f"Input:    {arr[0]!r}")
print(f"Result:   {result[0]!r}")
print(f"Expected: {'ﬁ test'.title()!r}")

print(f"\nInput dtype:  {arr.dtype}")
print(f"Result dtype: {result.dtype}")

print(f"\nInput length:    {len(arr[0])}")
print(f"Result length:   {len(result[0])}")
print(f"Expected length: {len('ﬁ test'.title())}")

# This assertion will fail due to truncation
assert result[0] == 'ﬁ test'.title(), f"Got {result[0]!r} but expected {'ﬁ test'.title()!r}"