# Bug Report: numpy.strings Null Character Truncation

**Target**: `numpy.strings` (functions: `strip`, `lstrip`, `rstrip`, `slice`, `rfind`)
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

NumPy string functions incorrectly truncate strings containing null characters (`\x00`), treating them as C-style string terminators rather than valid Unicode characters, which violates the documented behavior that these functions should behave like Python's string methods.

## Property-Based Test

```python
import numpy as np
import numpy.strings as ns
from hypothesis import given, strategies as st, settings


@given(st.lists(st.text(), min_size=1))
@settings(max_examples=1000)
def test_consistency_with_python_strip(strings):
    arr = np.array(strings)
    numpy_result = ns.strip(arr)

    for i, s in enumerate(strings):
        expected = s.strip()
        actual = numpy_result[i]
        assert expected == actual, f"Python: '{s}'.strip()='{expected}', NumPy: '{actual}'"


@given(st.lists(st.text(min_size=1), min_size=1), st.integers(min_value=0, max_value=10), st.integers(min_value=0, max_value=10))
@settings(max_examples=1000)
def test_slice_consistency_with_python(strings, start, stop):
    arr = np.array(strings)
    numpy_result = ns.slice(arr, start, stop)

    for i, s in enumerate(strings):
        expected = s[start:stop]
        actual = numpy_result[i]
        assert expected == actual, f"Python: '{s}'[{start}:{stop}]='{expected}', NumPy: '{actual}'"
```

<details>

<summary>
**Failing input**: `strings=['\x00']`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 31, in <module>
    test_consistency_with_python_strip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 7, in test_consistency_with_python_strip
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 15, in test_consistency_with_python_strip
    assert expected == actual, f"Python: '{s}'.strip()='{expected}', NumPy: '{actual}'"
           ^^^^^^^^^^^^^^^^^^
AssertionError: Python: ' '.strip()=' ', NumPy: ''
Falsifying example: test_consistency_with_python_strip(
    strings=['\x00'],
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.strings as ns

# Test string with null character at the end
s = 'ab\x00'
arr = np.array([s])

print("strip bug:")
print(f"  Python: '{s}'.strip() = {repr(s.strip())}")
print(f"  NumPy:  ns.strip(['{s}']) = {repr(ns.strip(arr)[0])}")

print("\nlstrip bug:")
print(f"  Python: '{s}'.lstrip() = {repr(s.lstrip())}")
print(f"  NumPy:  ns.lstrip(['{s}']) = {repr(ns.lstrip(arr)[0])}")

print("\nrstrip bug:")
print(f"  Python: '{s}'.rstrip() = {repr(s.rstrip())}")
print(f"  NumPy:  ns.rstrip(['{s}']) = {repr(ns.rstrip(arr)[0])}")

print("\nslice bug:")
print(f"  Python: '{s}'[0:10] = {repr(s[0:10])}")
print(f"  NumPy:  ns.slice(['{s}'], 0, 10) = {repr(ns.slice(arr, 0, 10)[0])}")

print("\nrfind bug:")
print(f"  Python: '{s}'.rfind('') = {s.rfind('')}")
print(f"  NumPy:  ns.rfind(['{s}'], '') = {ns.rfind(arr, '')[0]}")

# Test with just null character
s2 = '\x00'
arr2 = np.array([s2])

print("\n--- Testing with just '\\x00' ---")
print("strip bug:")
print(f"  Python: '{s2}'.strip() = {repr(s2.strip())}")
print(f"  NumPy:  ns.strip(['\\x00']) = {repr(ns.strip(arr2)[0])}")

# Test with null in the middle
s3 = 'ab\x00cd'
arr3 = np.array([s3])

print("\n--- Testing with 'ab\\x00cd' (null in middle) ---")
print("strip bug:")
print(f"  Python: '{s3}'.strip() = {repr(s3.strip())}")
print(f"  NumPy:  ns.strip(['ab\\x00cd']) = {repr(ns.strip(arr3)[0])}")

print("\nslice bug:")
print(f"  Python: '{s3}'[0:10] = {repr(s3[0:10])}")
print(f"  NumPy:  ns.slice(['ab\\x00cd'], 0, 10) = {repr(ns.slice(arr3, 0, 10)[0])}")
```

<details>

<summary>
Output shows null characters are truncated in NumPy
</summary>
```
strip bug:
  Python: 'ab '.strip() = 'ab\x00'
  NumPy:  ns.strip(['ab ']) = np.str_('ab')

lstrip bug:
  Python: 'ab '.lstrip() = 'ab\x00'
  NumPy:  ns.lstrip(['ab ']) = np.str_('ab')

rstrip bug:
  Python: 'ab '.rstrip() = 'ab\x00'
  NumPy:  ns.rstrip(['ab ']) = np.str_('ab')

slice bug:
  Python: 'ab '[0:10] = 'ab\x00'
  NumPy:  ns.slice(['ab '], 0, 10) = np.str_('ab')

rfind bug:
  Python: 'ab '.rfind('') = 3
  NumPy:  ns.rfind(['ab '], '') = 2

--- Testing with just '\x00' ---
strip bug:
  Python: ' '.strip() = '\x00'
  NumPy:  ns.strip(['\x00']) = np.str_('')

--- Testing with 'ab\x00cd' (null in middle) ---
strip bug:
  Python: 'ab cd'.strip() = 'ab\x00cd'
  NumPy:  ns.strip(['ab\x00cd']) = np.str_('ab\x00cd')

slice bug:
  Python: 'ab cd'[0:10] = 'ab\x00cd'
  NumPy:  ns.slice(['ab\x00cd'], 0, 10) = np.str_('ab\x00cd')
```
</details>

## Why This Is A Bug

This behavior violates the documented contract that NumPy string functions should behave like their Python counterparts. The numpy.strings module documentation explicitly references Python's string methods (e.g., "See Also: str.strip" in the docstring for `numpy.strings.strip`), establishing that these functions should produce identical results to Python's string methods when applied element-wise.

Python strings are sequences of Unicode code points and can contain null characters (`\x00`) as valid characters. The null character has no special meaning in Python strings - it's just Unicode code point U+0000. However, NumPy's string functions appear to treat `\x00` as a C-style string terminator, causing:

1. **Data loss**: Strings are silently truncated when null characters appear at the end
2. **Incorrect results**: Functions like `rfind` return wrong indices based on truncated string length
3. **Inconsistency**: Null characters in the middle of strings are preserved, but trailing nulls are removed

This affects any application that:
- Processes binary data represented as strings
- Works with text protocols that may include null bytes
- Expects NumPy to be a drop-in vectorized replacement for Python's string operations

## Relevant Context

The bug appears to stem from NumPy's internal C implementation treating strings as null-terminated. The Python-facing functions in `/numpy/_core/strings.py` delegate to C-level implementations (e.g., `_strip_whitespace`, `_strip_chars` from `numpy._core.umath`), which likely use C's string handling conventions.

Interestingly, null characters in the middle of strings are preserved correctly, suggesting the issue specifically affects how string lengths are determined when null appears at the end.

Documentation references:
- NumPy strings module: https://numpy.org/doc/stable/reference/routines.strings.html
- Source location: `/numpy/_core/strings.py` (lines 1050-1093 for `strip` function)

## Proposed Fix

The fix requires modifying NumPy's C-level string handling to use explicit length tracking rather than null-termination detection. Since this involves core C implementations, a high-level approach would be:

1. Ensure all string ufuncs in `numpy._core.umath` use the actual string length from the string dtype metadata rather than strlen-style null detection
2. Modify string storage/retrieval to preserve all bytes including trailing nulls
3. Add comprehensive test coverage for null character handling across all string functions

A complete fix would require changes to multiple C source files in NumPy's core, making it non-trivial to provide a simple patch. The issue fundamentally requires changing how NumPy's C code interprets string boundaries from null-terminated to length-delimited.