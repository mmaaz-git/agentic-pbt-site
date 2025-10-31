# Bug Report: numpy.char Null Byte Character Misinterpretation

**Target**: `numpy.char` string manipulation functions
**Severity**: High
**Bug Type**: Logic, Crash, Contract
**Date**: 2025-09-25

## Summary

String functions in `numpy.char` incorrectly treat the null byte character (`\x00`) as an empty string, causing wrong results in 8 functions and crashes in 3 functions. This violates the documented behavior that these functions call `str.method` element-wise.

## Property-Based Test

```python
import numpy as np
import numpy.char as char
from hypothesis import given, strategies as st, settings, example


@settings(max_examples=100)
@given(st.lists(st.text(alphabet=st.characters(blacklist_characters=['\x00']), min_size=1), min_size=1, max_size=5))
@example(['hello'])
def test_replace_null_byte_behavior(strings):
    """Test that numpy.char.replace handles null bytes correctly."""
    arr = np.array(strings, dtype=str)
    result = char.replace(arr, '\x00', 'X')

    for i, s in enumerate(strings):
        python_result = s.replace('\x00', 'X')
        assert result[i] == python_result, f"Failed for string {s!r}: got {result[i]!r}, expected {python_result!r}"


@settings(max_examples=100)
@given(st.lists(st.text(alphabet=st.characters(blacklist_characters=['\x00']), min_size=1), min_size=1, max_size=5))
@example(['hello'])
def test_count_null_byte_behavior(strings):
    """Test that numpy.char.count handles null bytes correctly."""
    arr = np.array(strings, dtype=str)
    result = char.count(arr, '\x00')

    for i, s in enumerate(strings):
        python_result = s.count('\x00')
        assert result[i] == python_result, f"Failed for string {s!r}: got {result[i]}, expected {python_result}"


@settings(max_examples=100)
@given(st.lists(st.text(alphabet=st.characters(blacklist_characters=['\x00']), min_size=1), min_size=1, max_size=5))
@example(['hello'])
def test_find_null_byte_behavior(strings):
    """Test that numpy.char.find handles null bytes correctly."""
    arr = np.array(strings, dtype=str)
    result = char.find(arr, '\x00')

    for i, s in enumerate(strings):
        python_result = s.find('\x00')
        assert result[i] == python_result, f"Failed for string {s!r}: got {result[i]}, expected {python_result}"


@settings(max_examples=100)
@given(st.lists(st.text(alphabet=st.characters(blacklist_characters=['\x00']), min_size=1), min_size=1, max_size=5))
@example(['hello'])
def test_rfind_null_byte_behavior(strings):
    """Test that numpy.char.rfind handles null bytes correctly."""
    arr = np.array(strings, dtype=str)
    result = char.rfind(arr, '\x00')

    for i, s in enumerate(strings):
        python_result = s.rfind('\x00')
        assert result[i] == python_result, f"Failed for string {s!r}: got {result[i]}, expected {python_result}"


if __name__ == "__main__":
    print("Testing null byte handling in numpy.char functions...")
    try:
        test_replace_null_byte_behavior()
        print("✓ test_replace_null_byte_behavior passed")
    except AssertionError as e:
        print(f"✗ test_replace_null_byte_behavior failed: {e}")

    try:
        test_count_null_byte_behavior()
        print("✓ test_count_null_byte_behavior passed")
    except AssertionError as e:
        print(f"✗ test_count_null_byte_behavior failed: {e}")

    try:
        test_find_null_byte_behavior()
        print("✓ test_find_null_byte_behavior passed")
    except AssertionError as e:
        print(f"✗ test_find_null_byte_behavior failed: {e}")

    try:
        test_rfind_null_byte_behavior()
        print("✓ test_rfind_null_byte_behavior passed")
    except AssertionError as e:
        print(f"✗ test_rfind_null_byte_behavior failed: {e}")
```

<details>

<summary>
**Failing input**: `['hello']` (any string without null bytes)
</summary>
```
Testing null byte handling in numpy.char functions...
✗ test_replace_null_byte_behavior failed: Failed for string 'hello': got np.str_('XhXeXlXlXoX'), expected 'hello'
✗ test_count_null_byte_behavior failed: Failed for string 'hello': got 6, expected 0
✗ test_find_null_byte_behavior failed: Failed for string 'hello': got 0, expected -1
✗ test_rfind_null_byte_behavior failed: Failed for string 'hello': got 5, expected -1
```
</details>

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
    result = char.index(arr, '\x00')
    print(f"index: returned {result[0]} (expected: ValueError)")
except ValueError as e:
    print(f"index: ValueError - {e} (correct)")

try:
    result = char.rindex(arr, '\x00')
    print(f"rindex: returned {result[0]} (expected: ValueError)")
except ValueError as e:
    print(f"rindex: ValueError - {e} (correct)")

print("\nCRASH BUGS:")
try:
    result = char.partition(arr, '\x00')
    print(f"partition: no crash, returned {result}")
except ValueError as e:
    print(f"partition: ValueError - {e}")

try:
    result = char.rpartition(arr, '\x00')
    print(f"rpartition: no crash, returned {result}")
except ValueError as e:
    print(f"rpartition: ValueError - {e}")

try:
    result = char.rsplit(arr, '\x00')
    print(f"rsplit: no crash, returned {result}")
except ValueError as e:
    print(f"rsplit: ValueError - {e}")
```

<details>

<summary>
ValueError crashes and incorrect return values
</summary>
```
LOGIC BUGS:
replace: np.str_('XhXeXlXlXoX') (expected: 'hello')
count: 6 (expected: 0)
find: 0 (expected: -1)
rfind: 5 (expected: -1)
startswith: True (expected: False)
endswith: True (expected: False)
index: returned 0 (expected: ValueError)
rindex: returned 5 (expected: ValueError)

CRASH BUGS:
partition: ValueError - empty separator
rpartition: ValueError - empty separator
rsplit: ValueError - empty separator
```
</details>

## Why This Is A Bug

This bug violates expected behavior in multiple critical ways:

1. **Contract Violation**: NumPy's documentation states that `numpy.char` functions "call `str.method` element-wise" on the array elements. However, the actual behavior contradicts Python's standard string methods:
   - Python: `'hello'.replace('\x00', 'X')` returns `'hello'` (no change, as there are no null bytes)
   - NumPy: `char.replace(['hello'], '\x00', 'X')` returns `'XhXeXlXlXoX'` (inserts X between every character)

2. **Null Byte vs Empty String Confusion**: The functions treat `\x00` (null byte, a valid Unicode character) as `''` (empty string):
   - Python correctly distinguishes: `'hello'.count('\x00')` = 0, but `'hello'.count('')` = 6
   - NumPy incorrectly conflates them: `char.count(['hello'], '\x00')` = 6 (same as empty string)

3. **Inconsistent Error Handling**: Some functions crash while Python's equivalents handle null bytes gracefully:
   - Python: `'hello'.partition('\x00')` returns `('hello', '', '')`
   - NumPy: `char.partition(['hello'], '\x00')` raises `ValueError: empty separator`

4. **Data Integrity Risk**: Applications using these functions to search for or replace null bytes will silently get completely wrong results, potentially leading to data corruption.

## Relevant Context

The root cause appears to be in NumPy's C string handling code where null bytes (`\x00`) are being incorrectly interpreted as C string terminators rather than valid characters within Python strings. This is a common issue when C code interfaces with Python strings, as C traditionally uses null-terminated strings while Python strings can contain null bytes and track their length separately.

The affected functions are part of the numpy.strings module (imported by numpy.char), which is implemented in C as part of the `_multiarray_umath` extension module. The bug affects at least 11 functions:
- Logic errors: `replace`, `count`, `find`, `rfind`, `index`, `rindex`, `startswith`, `endswith`
- Crashes: `partition`, `rpartition`, `rsplit`

NumPy documentation: https://numpy.org/doc/stable/reference/routines.char.html

## Proposed Fix

Since the implementation is in C extension code, a high-level fix approach would be:

1. **Locate the string argument processing** in NumPy's C string ufunc implementations (likely in `numpy/_core/src/umath/string_ufuncs.cpp` or similar)

2. **Preserve null bytes in Python strings** by:
   - Using PyUnicode APIs that respect the full string length rather than stopping at null bytes
   - Explicitly using `PyUnicode_GET_LENGTH()` instead of relying on null-termination
   - When converting to C strings, preserve the length information

3. **Add regression tests** to ensure null bytes are handled correctly across all string functions

4. **Special case empty string checks** to distinguish between `\x00` (length 1, contains null byte) and `''` (length 0, empty)

The fix requires modifying the C implementation to properly handle Python strings containing null bytes by using length-aware string operations rather than null-terminated C string functions.