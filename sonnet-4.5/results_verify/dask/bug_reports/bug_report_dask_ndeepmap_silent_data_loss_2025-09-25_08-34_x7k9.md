# Bug Report: dask.utils.ndeepmap Silent Data Loss

**Target**: `dask.utils.ndeepmap`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`ndeepmap(n, func, seq)` silently discards all but the first element when `n <= 0` and `seq` is a list with multiple elements. This can lead to silent data corruption as the function processes only one element while ignoring the rest.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
import dask.utils

@given(st.lists(st.integers(), min_size=2))
@settings(max_examples=100)
def test_ndeepmap_zero_depth_discards_elements(lst):
    inc = lambda x: x + 1
    result = dask.utils.ndeepmap(0, inc, lst)
    expected_if_only_first = inc(lst[0])
    assert result == expected_if_only_first
```

**Failing input**: Any list with more than one element, e.g., `[1, 2, 3]`

## Reproducing the Bug

```python
import dask.utils

inc = lambda x: x + 1

result1 = dask.utils.ndeepmap(0, inc, [1])
print(f"ndeepmap(0, inc, [1]) = {result1}")

result2 = dask.utils.ndeepmap(0, inc, [1, 2, 3, 4, 5])
print(f"ndeepmap(0, inc, [1, 2, 3, 4, 5]) = {result2}")

assert result1 == result2
```

**Output:**
```
ndeepmap(0, inc, [1]) = 2
ndeepmap(0, inc, [1, 2, 3, 4, 5]) = 2
```

The function returns `2` in both cases, silently discarding elements `[2, 3, 4, 5]`.

## Why This Is A Bug

The function silently discards data without any warning or error. When a user passes `[1, 2, 3, 4, 5]` expecting all elements to be processed, only the first element is used. This violates the principle of least surprise and can lead to silent data corruption.

The existing test in `dask/tests/test_utils.py` only tests with single-element lists:
```python
L = [1]
assert ndeepmap(0, inc, L) == 2
```

This masks the bug because a single-element list doesn't expose the discarding behavior.

## Fix

The function should either:
1. **Raise an error** when `n <= 0` and `seq` is a multi-element list (strictest fix)
2. **Apply the function to all elements** when `n <= 0` (most intuitive fix)
3. **Document the behavior** clearly if discarding is intentional (documentation fix)

### Option 1: Raise an error for invalid depth (Recommended)

```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -309,7 +309,11 @@ def ndeepmap(n, func, seq):
         return [func(item) for item in seq]
     elif n > 1:
         return [ndeepmap(n - 1, func, item) for item in seq]
     elif isinstance(seq, list):
-        return func(seq[0])
+        if len(seq) != 1:
+            raise ValueError(
+                f"ndeepmap with depth {n} expects a single-element list, got {len(seq)} elements"
+            )
+        return func(seq[0])
     else:
         return func(seq)
```

### Option 2: Apply to all elements (Alternative)

```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -309,7 +309,10 @@ def ndeepmap(n, func, seq):
         return [func(item) for item in seq]
     elif n > 1:
         return [ndeepmap(n - 1, func, item) for item in seq]
     elif isinstance(seq, list):
-        return func(seq[0])
+        if len(seq) == 1:
+            return func(seq[0])
+        else:
+            return [func(item) for item in seq]
     else:
         return func(seq)
```