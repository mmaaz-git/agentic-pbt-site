# Bug Report: numpy.char Null Byte Handling

**Target**: `numpy.char` (11 affected functions)
**Severity**: High
**Bug Type**: Logic + Crash
**Date**: 2025-09-25

## Summary

String functions in `numpy.char` incorrectly treat the null byte character (`\x00`) as an empty string, affecting 11 functions with wrong results or crashes.

## Affected Functions

**Logic bugs (wrong return values):**
- `replace`: Inserts replacement text between every character
- `count`: Returns `len(string) + 1` instead of `0`
- `find`: Returns `0` instead of `-1`
- `rfind`: Returns `len(string)` instead of `-1`
- `index`: Returns `0` instead of raising `ValueError`
- `rindex`: Returns `len(string)` instead of raising `ValueError`
- `startswith`: Returns `True` instead of `False`
- `endswith`: Returns `True` instead of `False`

**Crash bugs:**
- `partition`: Raises `ValueError: empty separator`
- `rpartition`: Raises `ValueError: empty separator`
- `rsplit`: Raises `ValueError: empty separator`

## Property-Based Test

```python
import numpy as np
import numpy.char as char
from hypothesis import given, strategies as st, settings


@settings(max_examples=500)
@given(st.lists(st.text(alphabet=st.characters(blacklist_characters=['\x00']), min_size=1), min_size=1))
def test_replace_null_byte_behavior(strings):
    arr = np.array(strings, dtype=str)
    result = char.replace(arr, '\x00', 'X')

    for i, s in enumerate(strings):
        python_result = s.replace('\x00', 'X')
        assert result[i] == python_result


@settings(max_examples=500)
@given(st.lists(st.text(alphabet=st.characters(blacklist_characters=['\x00']), min_size=1), min_size=1))
def test_count_null_byte_behavior(strings):
    arr = np.array(strings, dtype=str)
    result = char.count(arr, '\x00')

    for i, s in enumerate(strings):
        python_result = s.count('\x00')
        assert result[i] == python_result


@settings(max_examples=500)
@given(st.lists(st.text(alphabet=st.characters(blacklist_characters=['\x00']), min_size=1), min_size=1))
def test_find_null_byte_behavior(strings):
    arr = np.array(strings, dtype=str)
    result = char.find(arr, '\x00')

    for i, s in enumerate(strings):
        python_result = s.find('\x00')
        assert result[i] == python_result


@settings(max_examples=500)
@given(st.lists(st.text(alphabet=st.characters(blacklist_characters=['\x00']), min_size=1), min_size=1))
def test_rfind_null_byte_behavior(strings):
    arr = np.array(strings, dtype=str)
    result = char.rfind(arr, '\x00')

    for i, s in enumerate(strings):
        python_result = s.rfind('\x00')
        assert result[i] == python_result
```

**Failing input**: Any string without null bytes, e.g., `'hello'`, `'0'`, `'abc'`

## Reproducing the Bug

```python
import numpy as np
import numpy.char as char

arr = np.array(['hello'], dtype=str)

print("LOGIC BUGS:")
print(f"replace: {char.replace(arr, '\x00', 'X')[0]!r} (expected: 'hello')")
print(f"count: {char.count(arr, '\x00')[0]} (expected: 0)")
print(f"find: {char.find(arr, '\x00')[0]} (expected: -1)")
print(f"rfind: {char.rfind(arr, '\x00')[0]} (expected: -1)")
print(f"startswith: {char.startswith(arr, '\x00')[0]} (expected: False)")
print(f"endswith: {char.endswith(arr, '\x00')[0]} (expected: False)")

try:
    char.index(arr, '\x00')
    print(f"index: no exception (expected: ValueError)")
except ValueError:
    print(f"index: ValueError (correct)")

try:
    char.rindex(arr, '\x00')
    print(f"rindex: no exception (expected: ValueError)")
except ValueError:
    print(f"rindex: ValueError (correct)")

print("\nCRASH BUGS:")
try:
    char.partition(arr, '\x00')
except ValueError as e:
    print(f"partition: ValueError - {e}")

try:
    char.rpartition(arr, '\x00')
except ValueError as e:
    print(f"rpartition: ValueError - {e}")

try:
    char.rsplit(arr, '\x00')
except ValueError as e:
    print(f"rsplit: ValueError - {e}")
```

## Why This Is A Bug

The null byte (`\x00`) is a valid Unicode character, distinct from an empty string (`''`). Python's `str` methods correctly distinguish between them:

```python
'hello'.replace('\x00', 'X')  # 'hello' (no change)
'hello'.replace('', 'X')      # 'XhXeXlXlXoX' (inserts between chars)

'hello'.count('\x00')         # 0
'hello'.count('')             # 6 (len + 1)

'hello'.find('\x00')          # -1 (not found)
'hello'.startswith('\x00')    # False
'hello'.partition('\x00')     # ('', '', 'hello') (no crash)
```

NumPy's documentation states these functions "call `str.method` element-wise", but they deviate from Python's behavior when `\x00` is used as a search/replacement parameter. This is a contract violation that could lead to silent data corruption or unexpected crashes.

## Fix

The root cause is likely in NumPy's C string handling code where `\x00` is being incorrectly converted to an empty string. Since C strings are null-terminated, the null byte may be truncating the string argument.

The fix requires:

1. Locate the string argument processing in NumPy's string ufunc C code
2. Ensure null bytes within Python string objects are preserved, not interpreted as C string terminators
3. Use Python string length explicitly rather than relying on null-termination
4. Add tests to prevent regression

The likely location is in `numpy/_core/src/umath/string_ufuncs.cpp` or the string dtype handling code, where Python string arguments are converted to C representations.