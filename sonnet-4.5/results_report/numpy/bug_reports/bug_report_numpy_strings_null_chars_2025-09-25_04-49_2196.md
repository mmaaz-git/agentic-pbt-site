# Bug Report: numpy.strings Null Character Handling

**Target**: `numpy.strings` (multiple functions: `str_len`, `capitalize`, `find`, `slice`)
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Multiple functions in `numpy.strings` incorrectly handle null characters (`\x00`), treating them as C-style string terminators instead of valid Unicode characters. This affects `str_len`, `capitalize`, `find`, `slice`, and likely other functions.

## Property-Based Test

```python
import numpy as np
import numpy.strings as ns
from hypothesis import given, strategies as st, settings


null_char_texts = st.text(alphabet=st.sampled_from('abc\x00'), min_size=0, max_size=10)


@given(st.lists(null_char_texts, min_size=1, max_size=10))
@settings(max_examples=500)
def test_str_len_with_null_chars(string_list):
    arr = np.array(string_list)
    result = ns.str_len(arr)
    expected = np.array([len(s) for s in string_list])
    assert np.array_equal(result, expected), "str_len should count null characters"
```

**Failing input**: `string_list=['\x00']`

## Reproducing the Bug

```python
import numpy as np
import numpy.strings as ns

print("Bug 1: str_len")
arr = np.array(['\x00'])
print(f"  str_len(['\x00']): {ns.str_len(arr)[0]} (expected: 1)")

arr = np.array(['a\x00'])
print(f"  str_len(['a\x00']): {ns.str_len(arr)[0]} (expected: 2)")

print("\nBug 2: capitalize")
arr = np.array(['\x00'])
result = ns.capitalize(arr)
print(f"  capitalize(['\x00']): {repr(result[0])} (expected: '\x00')")

print("\nBug 3: find")
arr = np.array([''])
result = ns.find(arr, '\x00')
print(f"  find([''], '\x00'): {result[0]} (expected: -1)")

arr = np.array(['abc'])
result = ns.find(arr, '\x00')
print(f"  find(['abc'], '\x00'): {result[0]} (expected: -1)")

print("\nBug 4: slice")
arr = np.array(['\x000'])
result = ns.slice(arr, 0, 1)
print(f"  slice(['\x000'], 0, 1): {repr(result[0])} (expected: '\x00')")

arr = np.array(['a\x00b'])
result = ns.slice(arr, 0, 2)
print(f"  slice(['a\x00b'], 0, 2): {repr(result[0])} (expected: 'a\x00')")
```

## Why This Is A Bug

Python strings are length-prefixed and can contain null bytes anywhere as valid character data. All standard Python string methods handle null bytes correctly:

```python
>>> '\x00'.capitalize()
'\x00'
>>> len('\x00')
1
>>> 'a\x00b'[0:2]
'a\x00'
>>> ''.find('\x00')
-1
```

However, numpy.strings treats null bytes as C-style string terminators:
- `str_len(['\x00'])` returns 0 (stops at null)
- `str_len(['a\x00'])` returns 1 (stops at null)
- `capitalize(['\x00'])` returns empty string
- `find([''], '\x00')` returns 0 (false positive match)
- `slice` operations truncate at null characters

This is a fundamental issue affecting multiple functions, likely stemming from the use of C-string functions (like `strlen`, `strcpy`) in the underlying implementation instead of length-aware string operations.

## Fix

The fix requires updating the C/C++ implementation of numpy.strings to use length-aware string operations throughout. Specifically:
- Replace `strlen()` with explicit length tracking from Python string objects
- Replace null-terminated string functions with counted string operations
- Ensure all buffer operations use the string's actual length rather than scanning for null terminators
