# Bug Report: numpy.char.index/rindex Return Invalid Positions Instead of Raising ValueError for Null Byte Searches

**Target**: `numpy.char.index`, `numpy.char.rindex`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`numpy.char.index()` and `numpy.char.rindex()` fail to raise `ValueError` when searching for null bytes (`\x00`) in strings that don't contain them. Instead, they silently return incorrect position values, violating their documented behavior and Python's `str.index()` contract.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Property-based test using Hypothesis to find bugs in numpy.char.index and rindex
when searching for null bytes
"""

import numpy.char as char
from hypothesis import given, strategies as st, settings, assume

@given(st.text(min_size=0, max_size=30))
@settings(max_examples=1000)
def test_index_raises_for_null_when_not_found(s):
    assume('\x00' not in s)

    py_raised = False
    try:
        s.index('\x00')
    except ValueError:
        py_raised = True

    np_raised = False
    try:
        char.index(s, '\x00')
    except ValueError:
        np_raised = True

    assert py_raised == np_raised, f"index({repr(s)}, '\\x00'): Python raised={py_raised}, NumPy raised={np_raised}"

if __name__ == "__main__":
    # Run the test and let Hypothesis find a failing example
    test_index_raises_for_null_when_not_found()
```

<details>

<summary>
**Failing input**: `s=''` (or any string without null bytes)
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 31, in <module>
    test_index_raises_for_null_when_not_found()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 11, in test_index_raises_for_null_when_not_found
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 27, in test_index_raises_for_null_when_not_found
    assert py_raised == np_raised, f"index({repr(s)}, '\\x00'): Python raised={py_raised}, NumPy raised={np_raised}"
           ^^^^^^^^^^^^^^^^^^^^^^
AssertionError: index('', '\x00'): Python raised=True, NumPy raised=False
Falsifying example: test_index_raises_for_null_when_not_found(
    s='',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproducible example demonstrating numpy.char.index and rindex
incorrectly handling null byte searches
"""

import numpy.char as char

# Test string without null bytes
s = 'test'

print("="*60)
print("Test 1: numpy.char.index() with null byte search")
print("="*60)

print("String being searched: '{}'".format(s))
print("Searching for: '\\x00' (null byte)")
print()

print("Python str.index() behavior:")
try:
    result = s.index('\x00')
    print("  Result: {}".format(result))
except ValueError as e:
    print("  ValueError raised: {}".format(e))

print("\nnumpy.char.index() behavior:")
try:
    result = int(char.index(s, '\x00'))
    print("  Result: {}".format(result))
    print("  ERROR: Should have raised ValueError but returned {}!".format(result))
except ValueError as e:
    print("  ValueError raised: {}".format(e))

print("\n" + "="*60)
print("Test 2: numpy.char.rindex() with null byte search")
print("="*60)

print("String being searched: '{}'".format(s))
print("Searching for: '\\x00' (null byte)")
print()

print("Python str.rindex() behavior:")
try:
    result = s.rindex('\x00')
    print("  Result: {}".format(result))
except ValueError as e:
    print("  ValueError raised: {}".format(e))

print("\nnumpy.char.rindex() behavior:")
try:
    result = int(char.rindex(s, '\x00'))
    print("  Result: {}".format(result))
    print("  ERROR: Should have raised ValueError but returned {}!".format(result))
except ValueError as e:
    print("  ValueError raised: {}".format(e))

print("\n" + "="*60)
print("Test 3: Comparison with normal substring search")
print("="*60)

print("\nSearching for 'x' (which doesn't exist in 'test'):")
print("Python str.index('x'):")
try:
    result = s.index('x')
    print("  Result: {}".format(result))
except ValueError as e:
    print("  ValueError raised: {}".format(e))

print("\nnumpy.char.index('x'):")
try:
    result = int(char.index(s, 'x'))
    print("  Result: {}".format(result))
except ValueError as e:
    print("  ValueError raised: {}".format(e))

print("\n" + "="*60)
print("Summary")
print("="*60)
print("BUG: numpy.char.index() and rindex() fail to raise ValueError")
print("when searching for null bytes that don't exist in the string.")
print("Instead, they return invalid positions:")
print("  - index() returns 0 (suggesting null byte is at start)")
print("  - rindex() returns string length (suggesting null byte is at end)")
print("\nThis violates the documented behavior and Python's str.index() contract.")
```

<details>

<summary>
Incorrect position values returned instead of ValueError exception
</summary>
```
============================================================
Test 1: numpy.char.index() with null byte search
============================================================
String being searched: 'test'
Searching for: '\x00' (null byte)

Python str.index() behavior:
  ValueError raised: substring not found

numpy.char.index() behavior:
  Result: 0
  ERROR: Should have raised ValueError but returned 0!

============================================================
Test 2: numpy.char.rindex() with null byte search
============================================================
String being searched: 'test'
Searching for: '\x00' (null byte)

Python str.rindex() behavior:
  ValueError raised: substring not found

numpy.char.rindex() behavior:
  Result: 4
  ERROR: Should have raised ValueError but returned 4!

============================================================
Test 3: Comparison with normal substring search
============================================================

Searching for 'x' (which doesn't exist in 'test'):
Python str.index('x'):
  ValueError raised: substring not found

numpy.char.index('x'):
  ValueError raised: substring not found

============================================================
Summary
============================================================
BUG: numpy.char.index() and rindex() fail to raise ValueError
when searching for null bytes that don't exist in the string.
Instead, they return invalid positions:
  - index() returns 0 (suggesting null byte is at start)
  - rindex() returns string length (suggesting null byte is at end)

This violates the documented behavior and Python's str.index() contract.
```
</details>

## Why This Is A Bug

This is a clear violation of the documented API contract for `numpy.char.index()` and `numpy.char.rindex()`. According to the NumPy documentation and Python's string semantics:

1. **Documented behavior**: Both functions are documented as "Like `find`, but raises :exc:`ValueError` when the substring is not found." This promise is broken specifically for null byte searches.

2. **Inconsistent with Python's str methods**: Python's `str.index()` and `str.rindex()` correctly raise `ValueError` when searching for null bytes that don't exist in the string. NumPy's versions should behave identically to maintain compatibility.

3. **Returns misleading position values**:
   - `char.index()` returns 0, falsely indicating the null byte is at the beginning of the string
   - `char.rindex()` returns the string length, falsely indicating the null byte is at the end
   - Both positions are invalid since the null byte doesn't exist in the string content

4. **Breaks exception-based error handling**: Code that relies on catching `ValueError` to detect when a substring is not found will fail silently, potentially leading to incorrect program logic and hard-to-debug issues.

5. **Inconsistent behavior**: The functions correctly raise `ValueError` for all other non-existent substrings (e.g., searching for 'x' in 'test'), but fail only for null bytes, creating an unexpected special case.

## Relevant Context

This bug appears to stem from the underlying C implementation treating null bytes (`\x00`) as C-style string terminators rather than as searchable characters within the string content. The functions seem to be finding the implicit null terminator that exists at the end of C strings in memory, rather than searching within the actual Python string content.

The bug affects:
- NumPy version: 2.3.0
- Functions: `numpy.char.index()` and `numpy.char.rindex()`
- Only occurs when searching for null bytes (`\x00`)
- Affects all strings that don't contain null bytes (including empty strings)

Related NumPy source code locations:
- Python wrapper: `/numpy/_core/strings.py:353-383` (index) and `:387-417` (rindex)
- C implementation: Imported from `numpy._core.umath` as `_index_ufunc` and `_rindex_ufunc`

Documentation links:
- [numpy.char.index documentation](https://numpy.org/doc/stable/reference/generated/numpy.char.index.html)
- [numpy.char.rindex documentation](https://numpy.org/doc/stable/reference/generated/numpy.char.rindex.html)

## Proposed Fix

The bug requires fixing the C implementation to properly handle null bytes as searchable characters rather than string terminators. The functions should use length-aware string operations that don't rely on null termination.

```diff
--- a/numpy/_core/umath_string_impl.c
+++ b/numpy/_core/umath_string_impl.c
@@ index_implementation
-    // Current implementation likely uses C string functions that stop at \0
-    // For example, something like:
-    char *pos = strchr(str_ptr, search_char);
-    if (search_char == '\0') {
-        // Incorrectly returns position of C null terminator
-        return (pos != NULL) ? 0 : -1;  // Bug: returns 0 instead of raising
-    }
+    // Fixed: Use length-aware search that treats \0 as a regular character
+    for (size_t i = start; i < min(end, str_len); i++) {
+        if (str_ptr[i] == search_char) {
+            return i;
+        }
+    }
+    // Not found - signal to raise ValueError
+    return -1;
```

The fix should ensure that:
1. Null bytes are searched for within the actual string content (0 to str_len-1)
2. If not found, return a sentinel value that causes the Python wrapper to raise `ValueError`
3. The C null terminator at position str_len should not be considered part of the searchable string content