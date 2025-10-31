# Bug Report: numpy.strings.slice() Mishandles `None` as Stop Parameter

**Target**: `numpy.strings.slice()`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `stop=None` is explicitly passed to `numpy.strings.slice()`, the function incorrectly treats `start` as `stop`, returning only the first `start` characters instead of slicing from `start` to the end of the string.

## Property-Based Test

```python
import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st, example

@given(st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=10))
@example(['hello'])
def test_slice_with_none_stop(str_list):
    """Property: nps.slice(arr, start, None) should behave like Python arr[start:]"""
    arr = np.array(str_list, dtype='U')
    result = nps.slice(arr, 0, None)

    for i in range(len(arr)):
        expected = str_list[i][0:]
        assert result[i] == expected, f"slice(arr, 0, None) failed: got '{result[i]}', expected '{expected}'"
```

**Failing input**: `['hello']`

## Reproducing the Bug

```python
import numpy as np
import numpy.strings as nps

arr = np.array(['hello', 'world', 'test'])

result = nps.slice(arr, 1, None)
print(f"Result: {result}")
print(f"Expected: {np.array(['ello', 'orld', 'est'])}")
```

Output:
```
Result: ['h' 'w' 't']
Expected: ['ello' 'orld' 'est']
```

## Why This Is A Bug

The `numpy.strings.slice()` function is documented to work like Python slicing, where `s[start:None]` is equivalent to `s[start:]` (slice from start to end). However, when `stop=None` is explicitly passed, the function incorrectly invokes the special case mentioned in the documentation: "if only `start` is specified then it is interpreted as the `stop`".

This happens even though *both* `start` and `stop` were specified - the function treats explicit `None` the same as an unspecified parameter.

This violates:
1. Standard Python slicing semantics where `s[1:None] == s[1:]`
2. User expectations based on the documented behavior
3. The principle of least surprise - `None` should mean "no limit", not "unspecified"

## Fix

The bug is in how the function distinguishes between "parameter not provided" vs "parameter explicitly set to None". The function needs to check whether `stop` was actually provided as an argument, rather than just checking if `stop is None`.

A typical fix would involve using a sentinel value (like `Ellipsis` or a private object) as the default, so that `None` can be a legitimate value meaning "to the end":

```diff
-def slice(a, start=None, stop=None, step=None):
+_UNSPECIFIED = object()
+def slice(a, start=_UNSPECIFIED, stop=_UNSPECIFIED, step=_UNSPECIFIED):
     # Special case: if only start is provided, treat it as stop
-    if stop is None and step is None:
+    if stop is _UNSPECIFIED and step is _UNSPECIFIED:
         return a[:start]

     # Normal slicing
+    actual_start = None if start is _UNSPECIFIED else start
+    actual_stop = None if stop is _UNSPECIFIED else stop
+    actual_step = None if step is _UNSPECIFIED else step
-    return a[start:stop:step]
+    return a[actual_start:actual_stop:actual_step]
```

Note: The actual implementation may differ, but the key is distinguishing "not provided" from "explicitly None".