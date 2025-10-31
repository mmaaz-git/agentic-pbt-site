# Bug Report: numpy.char.str_len Incorrectly Truncates String Length at Trailing Null Characters

**Target**: `numpy.char.str_len`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.char.str_len()` returns incorrect string lengths when strings end with null characters (`\x00`), treating them as C-style string terminators instead of valid Unicode code points, contradicting both Python semantics and NumPy's own documentation.

## Property-Based Test

```python
import numpy.char as char
import numpy as np
from hypothesis import given, strategies as st, settings


@given(st.text(min_size=0, max_size=20))
@settings(max_examples=1000)
def test_str_len_matches_python_len(s):
    arr = np.array([s])
    numpy_len = char.str_len(arr)[0]
    python_len = len(s)
    assert numpy_len == python_len, f"numpy.char.str_len({s!r}) = {numpy_len}, but len({s!r}) = {python_len}"

if __name__ == "__main__":
    test_str_len_matches_python_len()
```

<details>

<summary>
**Failing input**: `'\x00'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 15, in <module>
    test_str_len_matches_python_len()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 7, in test_str_len_matches_python_len
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 12, in test_str_len_matches_python_len
    assert numpy_len == python_len, f"numpy.char.str_len({s!r}) = {numpy_len}, but len({s!r}) = {python_len}"
           ^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: numpy.char.str_len('\x00') = 0, but len('\x00') = 1
Falsifying example: test_str_len_matches_python_len(
    s='\x00',
)
```
</details>

## Reproducing the Bug

```python
import numpy.char as char
import numpy as np

test_cases = ['\x00', 'a\x00', 'abc\x00', '\x00\x00', 'abc\x00def']

print("Testing numpy.char.str_len with null characters:")
print("=" * 60)
print(f"{'String':20} | {'Python len':10} | {'numpy str_len':13} | {'Match?':7}")
print("-" * 60)

for s in test_cases:
    arr = np.array([s])
    numpy_len = char.str_len(arr)[0]
    python_len = len(s)
    match = "✓" if numpy_len == python_len else "✗"
    print(f'{repr(s):20} | {python_len:10} | {numpy_len:13} | {match:7}')

print("\nDetailed analysis:")
print("-" * 60)
print("Pattern: numpy.char.str_len() stops counting at trailing null characters")
print("but correctly handles null characters in the middle of strings.")
```

<details>

<summary>
String length mismatch for strings with trailing null characters
</summary>
```
Testing numpy.char.str_len with null characters:
============================================================
String               | Python len | numpy str_len | Match?
------------------------------------------------------------
'\x00'               |          1 |             0 | ✗
'a\x00'              |          2 |             1 | ✗
'abc\x00'            |          4 |             3 | ✗
'\x00\x00'           |          2 |             0 | ✗
'abc\x00def'         |          7 |             7 | ✓

Detailed analysis:
------------------------------------------------------------
Pattern: numpy.char.str_len() stops counting at trailing null characters
but correctly handles null characters in the middle of strings.
```
</details>

## Why This Is A Bug

The NumPy documentation for `numpy.char.str_len` explicitly states: "For Unicode strings, it is the number of Unicode code points." The null character `\x00` is a valid Unicode code point (U+0000 NULL), and Python strings correctly handle it as any other character.

The function exhibits inconsistent behavior:
1. **Correct**: When `\x00` appears in the middle of a string (e.g., `'abc\x00def'`), the function correctly counts it as a character and returns length 7.
2. **Incorrect**: When `\x00` appears at the end of a string (e.g., `'abc\x00'`), the function stops counting at the first trailing null, returning length 3 instead of 4.

This violates the documented contract that the function counts "Unicode code points" without exception. The behavior suggests the underlying C implementation incorrectly treats trailing nulls as C-style string terminators, despite Python strings not being null-terminated.

## Relevant Context

1. **Module Status**: The `numpy.char` module is marked as legacy in the NumPy documentation with potential for future deprecation. Users are advised to use `numpy.strings` instead. However, legacy code should still function correctly as documented.

2. **Python String Semantics**: In Python, `\x00` is a valid character that can appear anywhere in a string. Python's `len()` function correctly counts all characters including nulls. Since `numpy.char` methods are "based on the methods in the Python string module" (per documentation), they should exhibit Python-like behavior.

3. **Impact**: This bug affects:
   - Direct string length calculations for data containing null characters
   - String manipulation functions that rely on `str_len` internally (e.g., `center`, `ljust`, `rjust` with null fillchar)
   - Data validation and processing pipelines that depend on accurate string lengths

4. **Documentation Reference**: The function is documented at [numpy.strings.str_len](https://numpy.org/doc/stable/reference/generated/numpy.strings.str_len.html) and states it counts "Unicode code points" without any mention of special null character handling.

## Proposed Fix

The bug appears to be in the underlying C implementation that scans for null terminators instead of using the actual string length. The fix requires modifying the C implementation to use Python's string length directly:

```diff
--- a/numpy/_core/src/umath/string_ufuncs.c
+++ b/numpy/_core/src/umath/string_ufuncs.c
@@ -xxx,x +xxx,x @@
-    // Current implementation likely does something like:
-    size_t len = 0;
-    while (len < max_len && str[len] != '\0') {
-        len++;
-    }
+    // Fix: Use the actual Python string length
+    // For Unicode strings, get the proper length from the PyUnicode object
+    len = PyUnicode_GET_LENGTH(str_obj);
```

Alternatively, if the function must work with raw character arrays, it should be modified to not treat `\x00` as a terminator but as a valid character to count through the entire allocated string buffer.