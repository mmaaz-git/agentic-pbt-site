# Bug Report: numpy.strings.partition Truncates Separator

**Target**: `numpy.strings.partition`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `partition()` is called with a separator longer than the input string's dtype width, the separator is silently truncated to the input array's dtype width, causing incorrect partitioning behavior where the function incorrectly finds the truncated separator in the string.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
import numpy.strings as nps

text_no_null = st.text(alphabet=st.characters(blacklist_characters='\x00'), min_size=1, max_size=20)

@given(text_no_null, text_no_null.filter(lambda x: len(x) > 0))
@settings(max_examples=500)
def test_partition_matches_python(s, sep):
    arr = np.array([s])
    sep_arr = np.array([sep])

    part1, part2, part3 = nps.partition(arr, sep_arr)
    p1, p2, p3 = s.partition(sep)

    assert part1[0] == p1 and part2[0] == p2 and part3[0] == p3
```

**Failing input**: `s='0', sep='00'`

## Reproducing the Bug

```python
import numpy as np
import numpy.strings as nps

s = '0'
sep = '00'

arr = np.array([s])
sep_arr = np.array([sep])

part1, part2, part3 = nps.partition(arr, sep_arr)
expected = s.partition(sep)

print(f"Input: s={s!r}, sep={sep!r}")
print(f"Python result: {expected}")
print(f"NumPy result:  ({part1[0]!r}, {part2[0]!r}, {part3[0]!r})")

assert (part1[0], part2[0], part3[0]) == expected
```

Output:
```
Input: s='0', sep='00'
Python result: ('0', '', '')
NumPy result:  ('', '0', '')
AssertionError
```

## Why This Is A Bug

According to Python's `str.partition()` documentation and the function's docstring, when the separator is not found, the result should be `(string, '', '')` - the whole string in the first element and empty strings in the second and third.

However, when a separator longer than the input string's dtype is passed, NumPy casts it to the input array's dtype at line 1602:
```python
sep = sep.astype(a.dtype, copy=False)
```

For `arr=['0']` (dtype `<U1`) and `sep='00'`, the separator gets truncated to `'0'`. This truncated separator IS found in the string, causing incorrect partitioning: `('', '0', '')` instead of `('0', '', '')`.

## Fix

The separator should NOT be cast to the input array's dtype before searching. Instead, preserve the original separator dtype for the search operation, and only handle dtype conversion after determining the buffer sizes.

```diff
--- a/numpy/_core/strings.py
+++ b/numpy/_core/strings.py
@@ -1599,8 +1599,8 @@ def partition(a, sep):
     if np.result_type(a, sep).char == "T":
         return _partition(a, sep)

-    sep = sep.astype(a.dtype, copy=False)
     pos = _find_ufunc(a, sep, 0, MAX)
+    sep = sep.astype(a.dtype, copy=False)
     a_len = str_len(a)
     sep_len = str_len(sep)
```