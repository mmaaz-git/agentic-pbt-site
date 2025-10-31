# Bug Report: pandas.core.common.is_full_slice - Doesn't Recognize Canonical Full Slice

**Target**: `pandas.core.common.is_full_slice`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `is_full_slice` function incorrectly returns `False` for `slice(None, None)`, even though this is the canonical Python representation of a full slice and is semantically equivalent to `slice(0, length)`.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st
from pandas.core.common import is_full_slice


@given(st.integers(min_value=1, max_value=100))
def test_is_full_slice_should_recognize_slice_none_none(length):
    arr = np.arange(length)

    slice_explicit = slice(0, length)
    slice_canonical = slice(None, None)

    result_explicit = arr[slice_explicit]
    result_canonical = arr[slice_canonical]

    assert np.array_equal(result_explicit, result_canonical), \
        "slice(None, None) and slice(0, length) should select the same elements"

    is_full_explicit = is_full_slice(slice_explicit, length)
    is_full_canonical = is_full_slice(slice_canonical, length)

    assert is_full_explicit, "slice(0, length) should be recognized as a full slice"
    assert is_full_canonical, \
        f"slice(None, None) should be recognized as a full slice but got {is_full_canonical}"
```

**Failing input**: `length=1` (or any positive integer)

## Reproducing the Bug

```python
import numpy as np
from pandas.core.common import is_full_slice

length = 5
arr = np.arange(length)

slice_explicit = slice(0, length)
slice_canonical = slice(None, None)

print(f"arr[slice(0, {length})] = {arr[slice_explicit]}")
print(f"arr[slice(None, None)] = {arr[slice_canonical]}")
print(f"Results equal: {np.array_equal(arr[slice_explicit], arr[slice_canonical])}")
print()
print(f"is_full_slice(slice(0, {length}), {length}) = {is_full_slice(slice_explicit, length)}")
print(f"is_full_slice(slice(None, None), {length}) = {is_full_slice(slice_canonical, length)}")
```

Output:
```
arr[slice(0, 5)] = [0 1 2 3 4]
arr[slice(None, None)] = [0 1 2 3 4]
Results equal: True

is_full_slice(slice(0, 5), 5) = True
is_full_slice(slice(None, None), 5) = False
```

## Why This Is A Bug

The function `is_full_slice` is documented as checking "We have a full length slice." In Python, `slice(None, None)` (represented as `[:]` in slice notation) is the canonical and most common way to represent a full slice. It is semantically equivalent to `slice(0, length)` when applied to any sequence.

The current implementation only recognizes the explicit form `slice(0, length)` but not the canonical form `slice(None, None)`. This violates the principle of least surprise and could cause incorrect behavior in code that uses `is_full_slice` to check if a slice selects all elements.

## Fix

```diff
--- a/pandas/core/common.py
+++ b/pandas/core/common.py
@@ -xxx,x +xxx,x @@ def is_full_slice(obj, line: int) -> bool:
     """
     We have a full length slice.
     """
+    # Handle canonical full slice: slice(None, None, None)
+    if (isinstance(obj, slice) and
+        obj.start is None and
+        obj.stop is None and
+        obj.step is None):
+        return True
+
+    # Handle explicit full slice: slice(0, line, None)
     return (
         isinstance(obj, slice)
         and obj.start == 0
         and obj.stop == line
         and obj.step is None
     )
```