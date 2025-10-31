# Bug Report: numpy.strings.slice None End Parameter

**Target**: `numpy.strings.slice`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `numpy.strings.slice` function incorrectly handles the case where `stop=None` is explicitly passed with a `step` parameter, returning empty strings instead of slicing to the end of the string.

## Property-Based Test

```python
import numpy as np
import numpy.strings as ns
from hypothesis import given, strategies as st, settings


@given(
    st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=20),
    st.integers(min_value=0, max_value=5),
    st.integers(min_value=2, max_value=5)
)
@settings(max_examples=1000)
def test_slice_none_end_with_step(string_list, start, step):
    arr = np.array(string_list)
    result = ns.slice(arr, start, None, step)

    for i, s in enumerate(arr):
        expected = s[start:None:step]
        actual = result[i]
        assert actual == expected
```

**Failing input**: `string_list=['0'], start=0, step=2`

## Reproducing the Bug

```python
import numpy as np
import numpy.strings as ns

arr = np.array(["hello", "world"])

result = ns.slice(arr, 0, None, 2)
print("Result:", result)
print("Expected:", ["hlo", "wrd"])
```

## Why This Is A Bug

In Python, `"hello"[0:None:2]` correctly returns `"hlo"` (every 2nd character starting from index 0). However, `numpy.strings.slice(arr, 0, None, 2)` returns empty strings.

The root cause is in the swap logic at the beginning of the `slice` function:

```python
if stop is None:
    stop = start
    start = None
```

This logic is intended to handle the single-argument case (e.g., `slice(3)` → `slice(None, 3)`), but it incorrectly triggers when `stop=None` is explicitly passed with a non-None `step`. After the swap, when `start=0, stop=None, step=2` becomes `start=None, stop=0, step=2`, the subsequent logic sets `start=0` for positive step, resulting in a `[0:0:2]` slice which is empty.

## Fix

The swap logic should only apply when the function is called with a single positional argument (the "slice to position" case). When `step` is explicitly provided, the swap should not occur.

```diff
--- a/numpy/_core/strings.py
+++ b/numpy/_core/strings.py
@@ -1234,7 +1234,7 @@ def slice(a, start=None, stop=None, step=None, /):
     # Just like in the construction of a regular slice object, if only start
     # is specified then start will become stop, see logic in slice_new.
-    if stop is None:
+    if stop is None and step is None:
         stop = start
         start = None
```