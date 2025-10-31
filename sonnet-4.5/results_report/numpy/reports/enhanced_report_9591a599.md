# Bug Report: numpy.strings Functions Incorrectly Convert Null Bytes to Empty Strings

**Target**: `numpy.strings` (upper, lower, capitalize, title, swapcase, strip, partition)
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Multiple numpy.strings functions incorrectly convert standalone null byte characters (`\x00`) to empty strings, violating their documented behavior of mirroring Python's built-in string methods element-wise.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Hypothesis test demonstrating numpy.strings null byte handling bug"""

import numpy as np
import numpy.strings as ns
from hypothesis import given, strategies as st


@given(st.lists(st.just('\x00'), min_size=1))
def test_upper_preserves_null_bytes(strings):
    """Test that numpy.strings.upper preserves null bytes like Python's str.upper()"""
    arr = np.array(strings, dtype=np.str_)
    result = ns.upper(arr)

    for orig, res in zip(strings, result):
        expected = orig.upper()
        assert res == expected, f"Expected {repr(expected)}, got {repr(res)}"


if __name__ == "__main__":
    # Run the test
    test_upper_preserves_null_bytes()
```

<details>

<summary>
**Failing input**: `strings=['\x00']`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 22, in <module>
    test_upper_preserves_null_bytes()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 10, in test_upper_preserves_null_bytes
    def test_upper_preserves_null_bytes(strings):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 17, in test_upper_preserves_null_bytes
    assert res == expected, f"Expected {repr(expected)}, got {repr(res)}"
           ^^^^^^^^^^^^^^^
AssertionError: Expected '\x00', got np.str_('')
Falsifying example: test_upper_preserves_null_bytes(
    strings=['\x00'],  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
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
```

<details>

<summary>
All affected functions return empty strings instead of preserving null bytes
</summary>
```
Testing numpy.strings functions with null byte ('\x00'):
============================================================
upper:
  NumPy result: np.str_('')
  Python result: '\x00'
  Match: False

lower:
  NumPy result: np.str_('')
  Python result: '\x00'
  Match: False

capitalize:
  NumPy result: np.str_('')
  Python result: '\x00'
  Match: False

title:
  NumPy result: np.str_('')
  Python result: '\x00'
  Match: False

swapcase:
  NumPy result: np.str_('')
  Python result: '\x00'
  Match: False

strip:
  NumPy result: np.str_('')
  Python result: '\x00'
  Match: False

partition (with separator 'X'):
  NumPy result: (np.str_(''), np.str_(''), np.str_(''))
  Python result: ('\x00', '', '')
  Match: False

============================================================
Testing with null byte in middle of string ('hel\x00lo'):
upper: NumPy=np.str_('HEL\x00LO'), Python='HEL\x00LO'
Match: True
```
</details>

## Why This Is A Bug

This behavior violates the documented contract of numpy.strings functions, which explicitly state they "mirror Python's built-in str methods" and "call str.<method>() element-wise". The bug manifests as:

1. **Data Corruption**: Valid single-character strings containing null bytes are silently converted to empty strings, changing the length from 1 to 0 without warning.

2. **Documentation Contradiction**: The NumPy documentation (numpy.org/doc/stable) explicitly states these functions mirror Python's string methods. In Python:
   - `'\x00'.upper()` returns `'\x00'` (preserves the null byte)
   - `'\x00'` is a valid 1-character string
   - Case operations preserve string length and non-alphabetic characters

3. **Inconsistent Behavior**: The same functions correctly preserve null bytes when they appear within other characters (e.g., `'hel\x00lo'.upper()` correctly returns `'HEL\x00LO'`), but fail when the string consists solely of null bytes.

4. **Silent Failure**: No error or warning is raised when this data alteration occurs, making it difficult to detect in production code that processes binary data or protocols using null bytes as delimiters.

## Relevant Context

The issue affects NumPy version 2.3.0 and impacts at least the following functions:
- `numpy.strings.upper`
- `numpy.strings.lower`
- `numpy.strings.capitalize`
- `numpy.strings.title`
- `numpy.strings.swapcase`
- `numpy.strings.strip` (and likely `lstrip`, `rstrip`)
- `numpy.strings.partition` (and likely `rpartition`)

The bug appears to stem from C-level string handling that incorrectly treats standalone null bytes as string terminators, even though NumPy's string dtype should support null bytes as valid characters.

NumPy documentation references:
- https://numpy.org/doc/stable/reference/generated/numpy.strings.upper.html
- https://numpy.org/doc/stable/reference/routines.strings.html

## Proposed Fix

The fix requires modifying the C/ufunc implementation to properly handle strings containing null bytes. The implementation should:

1. Use explicit string length tracking rather than relying on null-termination
2. Ensure all string operations preserve the original string length for non-stripping operations
3. Treat `\x00` as a valid character throughout the processing pipeline

A high-level approach would be to modify the string processing functions in the NumPy C extensions to:
- Track string length explicitly using the array's dtype information
- Avoid using standard C string functions that stop at null terminators
- Ensure the output array preserves the full content including null bytes

The fix should ensure that `numpy.strings` functions produce identical results to Python's string methods for all valid Unicode strings, including those containing null bytes.