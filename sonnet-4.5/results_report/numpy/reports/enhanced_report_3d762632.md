# Bug Report: numpy.strings Functions Incorrectly Handle Null Characters

**Target**: `numpy.strings` (str_len, capitalize, find, slice)
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Multiple functions in `numpy.strings` incorrectly treat null characters (`\x00`) as C-style string terminators instead of valid Unicode characters, causing incorrect results, data truncation, and false positive matches.

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

if __name__ == "__main__":
    test_str_len_with_null_chars()
```

<details>

<summary>
**Failing input**: `string_list=['\x00']`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 18, in <module>
    test_str_len_with_null_chars()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 10, in test_str_len_with_null_chars
    @settings(max_examples=500)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 15, in test_str_len_with_null_chars
    assert np.array_equal(result, expected), "str_len should count null characters"
           ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
AssertionError: str_len should count null characters
Falsifying example: test_str_len_with_null_chars(
    string_list=['\x00'],
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.strings as ns

print("Bug 1: str_len")
arr = np.array(['\x00'])
print(f"  str_len(['\\x00']): {ns.str_len(arr)[0]} (expected: 1)")

arr = np.array(['a\x00'])
print(f"  str_len(['a\\x00']): {ns.str_len(arr)[0]} (expected: 2)")

print("\nBug 2: capitalize")
arr = np.array(['\x00'])
result = ns.capitalize(arr)
print(f"  capitalize(['\\x00']): {repr(result[0])} (expected: '\\x00')")

print("\nBug 3: find")
arr = np.array([''])
result = ns.find(arr, '\x00')
print(f"  find([''], '\\x00'): {result[0]} (expected: -1)")

arr = np.array(['abc'])
result = ns.find(arr, '\x00')
print(f"  find(['abc'], '\\x00'): {result[0]} (expected: -1)")

print("\nBug 4: slice")
arr = np.array(['\x000'])
result = ns.slice(arr, 0, 1)
print(f"  slice(['\\x000'], 0, 1): {repr(result[0])} (expected: '\\x00')")

arr = np.array(['a\x00b'])
result = ns.slice(arr, 0, 2)
print(f"  slice(['a\\x00b'], 0, 2): {repr(result[0])} (expected: 'a\\x00')")
```

<details>

<summary>
Multiple numpy.strings functions fail to handle null characters correctly
</summary>
```
Bug 1: str_len
  str_len(['\x00']): 0 (expected: 1)
  str_len(['a\x00']): 1 (expected: 2)

Bug 2: capitalize
  capitalize(['\x00']): np.str_('') (expected: '\x00')

Bug 3: find
  find([''], '\x00'): 0 (expected: -1)
  find(['abc'], '\x00'): 0 (expected: -1)

Bug 4: slice
  slice(['\x000'], 0, 1): np.str_('') (expected: '\x00')
  slice(['a\x00b'], 0, 2): np.str_('a') (expected: 'a\x00')
```
</details>

## Why This Is A Bug

This bug violates expected behavior in several critical ways:

1. **Documentation Contradiction**: The numpy.strings.str_len documentation explicitly states it returns "the number of Unicode code points" for Unicode strings. The null character (`\x00`, U+0000) is a valid Unicode code point and should be counted. The documentation makes no mention of null-terminated string behavior.

2. **Python Compatibility Violation**: The numpy.strings.capitalize documentation states it "Calls :meth:`str.capitalize` element-wise". Python's str.capitalize() correctly handles null bytes:
   ```python
   >>> '\x00'.capitalize()
   '\x00'
   >>> len('\x00')
   1
   ```
   NumPy's implementation contradicts this documented behavior.

3. **Data Integrity Loss**: The functions silently truncate data at null characters, causing data loss without warning. This is particularly serious for binary data or strings from external sources that may legitimately contain null bytes.

4. **False Positive Matches**: The find() function returns 0 (indicating a match at position 0) when searching for '\x00' in strings that don't contain it. This violates the documented behavior of returning -1 when the substring is not found, potentially causing incorrect program logic.

5. **Inconsistent with NumPy's Own Goals**: The numpy.strings module is meant to provide consistent, vectorized string operations based on Python's string methods. The current C-style null termination behavior fundamentally breaks this promise.

## Relevant Context

- **NumPy Version**: 2.3.0
- **Affected Functions**: At minimum str_len, capitalize, find, and slice are affected. Other string functions may have similar issues.
- **Root Cause**: The underlying implementation appears to use C string functions (like strlen, strcpy) that treat null bytes as string terminators, rather than using length-aware operations that respect Python's string semantics.
- **Python String Documentation**: Python's documentation explicitly states: "Since Python strings have an explicit length, %s conversions do not assume that '\0' is the end of the string." NumPy should follow this same principle.
- **Impact**: This affects any application processing binary data, interfacing with C libraries that may include null bytes in data, or handling strings from untrusted sources.

## Proposed Fix

The fix requires updating the C/C++ implementation of numpy.strings to use length-aware string operations. Here's the high-level approach:

1. Replace all uses of null-terminated C string functions with length-aware alternatives
2. Track string lengths explicitly from Python string objects
3. Use memcpy/memmove instead of strcpy for copying string data
4. Update comparison and search functions to use the actual string length rather than scanning for null terminators

The implementation would need to modify the core string handling in NumPy's C extension modules to properly handle the full string content including null bytes, similar to how Python's own string implementation works.