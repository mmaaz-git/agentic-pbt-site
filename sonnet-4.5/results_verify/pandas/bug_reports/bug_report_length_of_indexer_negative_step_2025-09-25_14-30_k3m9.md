# Bug Report: pandas.core.indexers.utils.length_of_indexer Incorrect Length for Negative Step Slices

**Target**: `pandas.core.indexers.utils.length_of_indexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `length_of_indexer` function returns negative lengths when computing the length of slices with negative steps and `None` for start/stop values (e.g., `slice(None, None, -1)`). This causes incorrect validation in `check_setitem_lengths` and violates the function's documented contract of returning the expected length.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.core.indexers.utils import length_of_indexer


@given(
    st.one_of(
        st.integers(min_value=0, max_value=100),
        st.builds(
            slice,
            st.one_of(st.none(), st.integers(min_value=-50, max_value=50)),
            st.one_of(st.none(), st.integers(min_value=-50, max_value=50)),
            st.one_of(st.none(), st.integers(min_value=-10, max_value=10).filter(lambda x: x != 0)),
        ),
        st.lists(st.integers(min_value=0, max_value=100), min_size=0, max_size=50),
        st.lists(st.booleans(), min_size=1, max_size=50),
    ),
    st.integers(min_value=1, max_value=100)
)
@settings(max_examples=1000)
def test_length_of_indexer_matches_actual_length(indexer, target_len):
    target = list(range(target_len))

    if isinstance(indexer, list) and len(indexer) > 0 and isinstance(indexer[0], bool):
        if len(indexer) != target_len:
            return

    try:
        claimed_length = length_of_indexer(indexer, target)
        actual_result = target[indexer]
        actual_length = len(actual_result)
        assert claimed_length == actual_length
    except (IndexError, ValueError, TypeError, KeyError):
        pass
```

**Failing input**: `indexer=slice(None, None, -1), target_len=1`

## Reproducing the Bug

```python
from pandas.core.indexers.utils import length_of_indexer

target = [0]
indexer = slice(None, None, -1)

claimed_length = length_of_indexer(indexer, target)
actual_length = len(target[indexer])

print(f"Expected: {actual_length}")
print(f"Got: {claimed_length}")
```

**Output:**
```
Expected: 1
Got: -1
```

## Why This Is A Bug

The function's docstring states it returns "the expected length of target[indexer]". Lengths cannot be negative, yet the function returns -1 for the common pattern `slice(None, None, -1)` (reverse slice).

This affects:
1. The `check_setitem_lengths` validation function that uses this to validate assignments
2. Any code relying on accurate length calculations for negative-step slices

The root cause is that when `start` and `stop` are `None`, the code defaults them to `0` and `target_len` respectively (lines 303-308), assuming a positive step. When the step is negative, it then tries to convert these values (line 314), but this conversion is incorrect for the `None` case.

## Fix

The fix is to handle `None` values for start/stop differently based on the sign of step, before applying the conversion logic:

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -298,18 +298,28 @@ def length_of_indexer(indexer, target=None) -> int:
     if target is not None and isinstance(indexer, slice):
         target_len = len(target)
         start = indexer.start
         stop = indexer.stop
         step = indexer.step
-        if start is None:
-            start = 0
-        elif start < 0:
-            start += target_len
-        if stop is None or stop > target_len:
-            stop = target_len
-        elif stop < 0:
-            stop += target_len
+
         if step is None:
             step = 1
         elif step < 0:
+            # For negative step, None values have different defaults
+            if start is None:
+                start = target_len - 1
+            elif start < 0:
+                start += target_len
+            if stop is None:
+                stop = -target_len - 1
+            elif stop < 0:
+                stop += target_len
             start, stop = stop + 1, start + 1
             step = -step
+        else:
+            # For positive step (or when step was None)
+            if start is None:
+                start = 0
+            elif start < 0:
+                start += target_len
+            if stop is None or stop > target_len:
+                stop = target_len
+            elif stop < 0:
+                stop += target_len
         return (stop - start + step - 1) // step
```