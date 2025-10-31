# Bug Report: numpy.char.index/rindex Don't Raise ValueError for Null Byte Searches

**Target**: `numpy.char.index`, `numpy.char.rindex`
**Severity**: High
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`numpy.char.index()` and `numpy.char.rindex()` fail to raise `ValueError` when searching for null bytes that don't exist in strings. Instead, they return invalid positions (0 for index, string length for rindex), violating Python's `str.index()` contract.

## Property-Based Test

```python
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
```

**Failing input**: Any string without null bytes, e.g., `s='test'`

## Reproducing the Bug

```python
import numpy.char as char

s = 'test'

print("Python str.index:")
try:
    result = s.index('\x00')
    print(f"  Result: {result}")
except ValueError as e:
    print(f"  ValueError: {e}")

print("\nnumpy.char.index:")
try:
    result = int(char.index(s, '\x00'))
    print(f"  Result: {result}")
except ValueError as e:
    print(f"  ValueError: {e}")

print("\n---")

print("\nPython str.rindex:")
try:
    result = s.rindex('\x00')
    print(f"  Result: {result}")
except ValueError as e:
    print(f"  ValueError: {e}")

print("\nnumpy.char.rindex:")
try:
    result = int(char.rindex(s, '\x00'))
    print(f"  Result: {result}")
except ValueError as e:
    print(f"  ValueError: {e}")
```

Output:
```
Python str.index:
  ValueError: substring not found

numpy.char.index:
  Result: 0

---

Python str.rindex:
  ValueError: substring not found

numpy.char.rindex:
  Result: 4
```

## Why This Is A Bug

1. **Violates API contract**: `index()` and `rindex()` are documented to raise `ValueError` when the substring is not found, but they return positions instead when searching for null bytes.

2. **Returns invalid positions**:
   - `index()` returns 0, suggesting null byte is at the start
   - `rindex()` returns the string length, suggesting null byte is at the end
   - Both are incorrect - the null byte doesn't exist in the string

3. **Breaks error handling**: Code that catches `ValueError` to detect missing substrings will fail, as the exception is never raised for null byte searches.

4. **Related to find bug**: This is consistent with `char.find()` returning 0 instead of -1 for null bytes - both stem from treating null bytes as C-style string terminators.

## Fix

The bug is in the C implementation treating null bytes as string terminators. The functions appear to be finding the implicit null terminator that exists in C strings:
- `index` finds it at the "start" (position 0 in the conceptual sense)
- `rindex` finds it at the "end" (the actual C null terminator position)

The fix requires:
- Using length-aware string operations that don't rely on null termination
- Properly searching for null bytes within the actual string content
- Raising ValueError when null bytes are not found in the string content, not when encountering the C null terminator

```diff
--- a/numpy/_core/strings.py
+++ b/numpy/_core/strings.py
@@ -index_implementation
-    # Current: finds C null terminator when searching for \x00
+    # Fixed: search within actual string length
     pos = find_in_buffer(str, substr, start, end, str_len)
     if pos < 0:
         raise ValueError("substring not found")
     return pos
```