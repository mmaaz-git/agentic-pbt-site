# Bug Report: numpy.char Null Character Handling

**Target**: `numpy.char` functions: find, rfind, count, index, rindex, startswith, endswith, replace
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Multiple numpy.char functions (find, rfind, count, index, rindex, startswith, endswith, replace) produce incorrect results when operating on the null character '\x00', treating it as a string terminator rather than a valid Unicode character.

## Property-Based Test

```python
import numpy as np
import numpy.char
from hypothesis import given, settings, strategies as st


@given(st.lists(st.text(min_size=1), min_size=1), st.text(min_size=1, max_size=5))
@settings(max_examples=1000)
def test_find_matches_python(string_list, substring):
    arr = np.array(string_list)
    np_result = numpy.char.find(arr, substring)

    for i, s in enumerate(arr):
        py_result = s.find(substring)
        assert np_result[i] == py_result


@given(st.lists(st.text(min_size=1), min_size=1), st.text(min_size=1, max_size=5))
@settings(max_examples=1000)
def test_count_matches_python(string_list, substring):
    arr = np.array(string_list)
    np_result = numpy.char.count(arr, substring)

    for i, s in enumerate(arr):
        py_result = s.count(substring)
        assert np_result[i] == py_result


@given(st.lists(st.text(min_size=1), min_size=1))
@settings(max_examples=1000)
def test_startswith_matches_python(string_list):
    arr = np.array(string_list)
    np_result = numpy.char.startswith(arr, '\x00')

    for i, s in enumerate(arr):
        py_result = s.startswith('\x00')
        assert np_result[i] == py_result


@given(st.lists(st.text(min_size=1), min_size=1))
@settings(max_examples=1000)
def test_replace_matches_python(string_list):
    arr = np.array(string_list)
    np_result = numpy.char.replace(arr, '\x00', 'X')

    for i, s in enumerate(arr):
        py_result = s.replace('\x00', 'X')
        assert np_result[i] == py_result
```

**Failing input**: string='0', substring='\x00'

## Reproducing the Bug

```python
import numpy as np
import numpy.char

arr = np.array(['0'])
substring = '\x00'

np_find = numpy.char.find(arr, substring)[0]
py_find = '0'.find(substring)
print(f"find:   numpy={np_find}, python={py_find}")

np_rfind = numpy.char.rfind(arr, substring)[0]
py_rfind = '0'.rfind(substring)
print(f"rfind:  numpy={np_rfind}, python={py_rfind}")

np_count = numpy.char.count(arr, substring)[0]
py_count = '0'.count(substring)
print(f"count:  numpy={np_count}, python={py_count}")

try:
    np_index = numpy.char.index(arr, substring)[0]
    print(f"index:  numpy={np_index} (should raise ValueError!)")
except ValueError:
    print(f"index:  numpy raised ValueError")

arr2 = np.array(['a\x00b'])
print(f"\nWith null in string:")
print(f"find:   numpy={numpy.char.find(arr2, substring)[0]}, python={'a\x00b'.find(substring)}")
print(f"count:  numpy={numpy.char.count(arr2, substring)[0]}, python={'a\x00b'.count(substring)}")

print("\nstartswith/endswith bugs:")
arr3 = np.array(['hello', '\x00test', 'test\x00'])
for s in arr3:
    np_starts = numpy.char.startswith(np.array([s]), '\x00')[0]
    py_starts = s.startswith('\x00')
    print(f"  {repr(s)}: startswith numpy={np_starts}, python={py_starts}")

print("\nreplace bug:")
arr4 = np.array(['hello'])
np_replace = numpy.char.replace(arr4, '\x00', 'X')[0]
py_replace = 'hello'.replace('\x00', 'X')
print(f"  replace('\\x00', 'X'): numpy={repr(np_replace)}, python={repr(py_replace)}")

print("\npartition bug:")
try:
    numpy.char.partition(np.array(['a\x00b']), '\x00')
    print("  partition succeeded")
except ValueError as e:
    print(f"  partition raised: {e}")

try:
    'a\x00b'.partition('\x00')
    print("  python partition succeeded")
except ValueError as e:
    print(f"  python partition raised: {e}")
```

Output:
```
find:   numpy=0, python=-1
rfind:  numpy=1, python=-1
count:  numpy=2, python=0
index:  numpy=0 (should raise ValueError!)

With null in string:
find:   numpy=0, python=1
count:  numpy=4, python=1

startswith/endswith bugs:
  'hello': startswith numpy=True, python=False
  '\x00test': startswith numpy=True, python=True
  'test\x00': startswith numpy=True, python=False

replace bug:
  replace('\x00', 'X'): numpy='XhXeXlXlXoX', python='hello'

partition bug:
  partition raised: empty separator
  python partition succeeded
```

## Why This Is A Bug

The numpy.char documentation states these functions call the corresponding Python str methods element-wise. However, when operating on '\x00':

**Search functions** (find, rfind, count, index, rindex):
- `find` returns 0 instead of -1 when '\x00' is not in the string
- `rfind` returns string length instead of -1 when '\x00' is not in the string
- `count` returns string length + 1 instead of 0 or the correct count
- `index` returns 0 instead of raising ValueError when '\x00' is not found
- Even when '\x00' IS present, all return incorrect positions/counts

**Prefix/suffix functions** (startswith, endswith):
- `startswith('\x00')` returns True for ALL strings, even those not starting with '\x00'
- `endswith('\x00')` returns True for ALL strings, even those not ending with '\x00'

**Replace function**:
- `replace('\x00', 'X')` replaces every character boundary (before/after each char) instead of just '\x00' occurrences
- For 'hello', it produces 'XhXeXlXlXoX' instead of 'hello'

**Partition/split functions** (partition, rpartition, split):
- All raise ValueError: "empty separator" when using '\x00' as separator
- Python's str.partition/rpartition/split correctly handle '\x00' as a valid separator

The root cause appears to be C-style string handling that treats '\x00' as a terminator or empty string. This violates documented behavior and causes:
- Out-of-bounds indices that could crash downstream code
- Silent logical errors in string processing
- Violation of the ValueError contract for index/rindex
- Incorrect substring matching and counting

## Fix

The bug exists in the underlying C implementation. The code likely uses C-style string functions that treat '\x00' as a terminator. The fix requires updating the implementation to:

1. Use length-aware string operations instead of null-terminated C strings
2. Properly handle embedded null characters as valid Unicode code points
3. Ensure all five functions correctly match Python's str behavior for all valid Unicode characters including '\x00'

The relevant C/C++ implementations need to be audited and fixed, including:
- `_find_ufunc`, `_rfind_ufunc` for search operations
- count, startswith, endswith implementations
- replace/substitute implementations
- partition, rpartition, split implementations

All should use length-aware string operations that properly handle embedded null characters as valid Unicode code points.