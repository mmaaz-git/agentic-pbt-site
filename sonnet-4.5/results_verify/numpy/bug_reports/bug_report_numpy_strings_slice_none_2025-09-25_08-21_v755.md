# Bug Report: numpy.strings.slice Explicit None Parameter Handling

**Target**: `numpy.strings.slice`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `None` is explicitly passed as the `stop` parameter to `numpy.strings.slice(a, start, None)`, the function incorrectly treats it as a single-argument call, interpreting the `start` as `stop`, instead of treating `None` as "end of string" per Python slice semantics.

## Property-Based Test

```python
import numpy as np
import numpy.strings as ns
from hypothesis import given, strategies as st


@given(st.lists(st.text(min_size=1), min_size=1), st.integers(min_value=-20, max_value=20))
def test_slice_with_none_stop(strings, start):
    arr = np.array(strings, dtype=np.str_)
    result = ns.slice(arr, start, None)

    for orig, sliced_val in zip(strings, result):
        expected = orig[start:None]
        assert sliced_val == expected
```

**Failing input**: `strings=['hello'], start=1`

## Reproducing the Bug

```python
import numpy as np
import numpy.strings as ns

arr = np.array(['hello'], dtype=np.str_)

result = ns.slice(arr, 1, None)
expected = 'hello'[1:None]

print(f"ns.slice(['hello'], 1, None) = '{result[0]}'")
print(f"Expected: '{expected}'")
```

**Output:**
```
ns.slice(['hello'], 1, None) = 'h'
Expected: 'ello'
```

## Why This Is A Bug

The implementation cannot distinguish between:
1. `ns.slice(arr, 1)` - single argument where user wants `arr[:1]`
2. `ns.slice(arr, 1, None)` - two arguments where user wants `arr[1:None]` = `arr[1:]`

The source code contains:
```python
if stop is None:
    stop = start
    start = None
```

This swaps parameters when `stop is None`, which is correct for case 1 but wrong for case 2. Python's `slice(1, None)` means "from index 1 to end".

## Fix

```diff
--- a/numpy/_core/strings.py
+++ b/numpy/_core/strings.py
@set_module("numpy.strings")
-def slice(a, start=None, stop=None, step=None, /):
+def slice(a, *args):
+    if len(args) == 1:
+        start, stop, step = None, args[0], None
+    elif len(args) == 2:
+        start, stop, step = args[0], args[1], None
+    elif len(args) == 3:
+        start, stop, step = args[0], args[1], args[2]
+    else:
+        start, stop, step = None, None, None
+
-    if stop is None:
-        stop = start
-        start = None
```