# Bug Report: numpy.char.replace Catastrophically Mishandles Null Bytes

**Target**: `numpy.char.replace`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.char.replace()` has catastrophic bugs when handling null bytes (`\x00`), producing completely incorrect results that bear no resemblance to Python's `str.replace()` behavior. This causes severe data corruption.

## Property-Based Test

```python
import numpy.char as char
from hypothesis import given, strategies as st, settings

@given(st.text(min_size=0, max_size=20), st.text(min_size=1, max_size=5), st.text(min_size=0, max_size=5))
@settings(max_examples=1000)
def test_replace_matches_python(s, old, new):
    py_result = s.replace(old, new)
    np_result = str(char.replace(s, old, new))

    assert py_result == np_result, f"replace({repr(s)}, {repr(old)}, {repr(new)}): Python={repr(py_result)}, NumPy={repr(np_result)}"
```

**Failing input**: `s='test'`, `old='\x00'`, `new='X'`

## Reproducing the Bug

```python
import numpy.char as char

print("Bug 1: Null byte as 'old' parameter")
s = 'test'
result = str(char.replace(s, '\x00', 'X'))
print(f"char.replace('test', '\\x00', 'X') = {repr(result)}")
print(f"Expected (Python): {repr('test')}")
print(f"Actual (NumPy):    {repr(result)}")

print("\nBug 2: Removing null bytes")
s = 'te\x00st'
result = str(char.replace(s, '\x00', ''))
print(f"char.replace('te\\x00st', '\\x00', '') = {repr(result)}")
print(f"Expected (Python): {repr('test')}")
print(f"Actual (NumPy):    {repr(result)}")

print("\nBug 3: Inserting null bytes")
s = 'test'
result = str(char.replace(s, 'e', '\x00'))
print(f"char.replace('test', 'e', '\\x00') = {repr(result)}")
print(f"Expected (Python): {repr('t\x00st')}")
print(f"Actual (NumPy):    {repr(result)}")
```

Output:
```
Bug 1: Null byte as 'old' parameter
char.replace('test', '\x00', 'X') = 'XtXeXsXtX'
Expected (Python): 'test'
Actual (NumPy):    'XtXeXsXtX'

Bug 2: Removing null bytes
char.replace('te\x00st', '\x00', '') = 'te\x00st'
Expected (Python): 'test'
Actual (NumPy):    'te\x00st'

Bug 3: Inserting null bytes
char.replace('test', 'e', '\x00') = 'tst'
Expected (Python): 't\x00st'
Actual (NumPy):    'tst'
```

## Why This Is A Bug

1. **Severe data corruption**: When null byte is used as the `old` parameter, the function inserts the replacement between **every character**, producing completely incorrect output ('test' → 'XtXeXsXtX').

2. **Cannot remove null bytes**: When trying to replace null bytes with empty string, the function fails to perform the replacement, leaving the null bytes in the string.

3. **Strips null bytes from replacements**: When null bytes are in the `new` parameter, they get stripped from the result ('test'.replace('e', '\x00') should give 't\x00st' but gives 'tst').

4. **Violates documented contract**: The docstring claims to call `str.replace` element-wise, but produces completely different results.

5. **Silent failures**: The function doesn't raise errors; it silently produces wrong results, making debugging difficult.

## Fix

The issue appears to be in how the C implementation handles null bytes. When a null byte is used as the search pattern, it's being treated as a match at every position (like an empty string in Python's replace), but the implementation is buggy.

The function needs to properly handle null bytes as regular characters:
- Check actual string length, not rely on null termination
- Use length-aware string operations instead of C-style null-terminated operations
- Ensure null bytes in both `old` and `new` parameters are handled correctly