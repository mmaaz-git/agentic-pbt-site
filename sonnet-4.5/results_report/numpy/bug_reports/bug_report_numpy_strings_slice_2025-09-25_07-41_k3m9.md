# Bug Report: numpy.strings.slice with stop=None

**Target**: `numpy.strings.slice`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `numpy.strings.slice` is called with `stop=None` explicitly, it incorrectly treats it as if only `start` was specified, causing the start and stop parameters to be swapped. This results in empty or truncated strings instead of slicing to the end.

## Property-Based Test

```python
import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st, settings

@st.composite
def string_arrays(draw):
    str_list = draw(st.lists(
        st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=1, max_size=50),
        min_size=1, max_size=20
    ))
    return np.array(str_list, dtype='U')

@settings(max_examples=200)
@given(string_arrays(), st.integers(min_value=0, max_value=10))
def test_slice_with_none_stop_matches_python(arr, start):
    result = nps.slice(arr, start, None)

    for i in range(len(arr)):
        s = str(arr[i])
        start_val = min(start, len(s))
        expected = s[start_val:]
        actual = str(result[i])
        assert actual == expected
```

**Failing input**: `arr=array(['0'], dtype='<U1')`, `start=0`

## Reproducing the Bug

```python
import numpy as np
import numpy.strings as nps

arr = np.array(['hello'], dtype='U')
result = nps.slice(arr, 0, None)
print(result)

s = 'hello'
print(s[0:None])
```

Expected output:
```
['hello']
hello
```

Actual output:
```
['']
hello
```

## Why This Is A Bug

Python's slicing semantics treat `s[0:None]` as `s[0:]` (slice from index 0 to end). The numpy.strings.slice function should match this behavior when `stop=None` is explicitly passed.

However, the current implementation cannot distinguish between:
1. `slice(a, 0)` - should become `a[:0]` (empty) per the "start-only becomes stop" rule
2. `slice(a, 0, None)` - should become `a[0:]` (full slice from 0 to end)

Both cases result in `start=0, stop=None` after parameter binding, and the implementation incorrectly swaps them.

## Fix

The function should use a sentinel value to detect when `stop` is truly omitted versus explicitly set to `None`. Here's a suggested fix:

```diff
--- a/numpy/strings/__init__.py
+++ b/numpy/strings/__init__.py
@@ -xxx,x +xxx,x @@ def slice(a, start=None, stop=None, step=None, /):
+_SLICE_SENTINEL = object()
+
 @set_module("numpy.strings")
-def slice(a, start=None, stop=None, step=None, /):
+def slice(a, start=None, stop=_SLICE_SENTINEL, step=None, /):
     """
     Slice the strings in `a` by slices specified by `start`, `stop`, `step`.
     Like in the regular Python `slice` object, if only `start` is
     specified then it is interpreted as the `stop`.

     [rest of docstring...]
     """
-    if stop is None:
+    if stop is _SLICE_SENTINEL:
         stop = start
         start = None
+
+    if stop is None:
+        stop = np.where(step < 0, np.iinfo(np.intp).min, np.iinfo(np.intp).max)
```

This way:
- `slice(a, 2)` → `stop=_SLICE_SENTINEL` → swaps to `a[:2]` ✓
- `slice(a, 0, None)` → `stop=None` → becomes `a[0:]` ✓