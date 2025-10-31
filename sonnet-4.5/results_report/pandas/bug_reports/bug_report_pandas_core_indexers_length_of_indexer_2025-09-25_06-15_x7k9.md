# Bug Report: pandas.core.indexers length_of_indexer negative step

**Target**: `pandas.core.indexers.length_of_indexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `length_of_indexer` function incorrectly calculates the expected length for slices with negative steps when `start` or `stop` are `None`. This results in returning negative lengths or incorrect values instead of the actual slice length.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.core.indexers import length_of_indexer

@given(
    start=st.integers(min_value=-100, max_value=100) | st.none(),
    stop=st.integers(min_value=-100, max_value=100) | st.none(),
    step=st.integers(min_value=-10, max_value=10).filter(lambda x: x != 0) | st.none(),
    target_len=st.integers(min_value=0, max_value=100),
)
def test_length_of_indexer_matches_actual_length(start, stop, step, target_len):
    slc = slice(start, stop, step)
    target = list(range(target_len))

    expected_length = length_of_indexer(slc, target)
    actual_length = len(target[slc])

    assert expected_length == actual_length
```

**Failing input**: `slice(None, None, -1)` with `target_len=1`

## Reproducing the Bug

```python
from pandas.core.indexers import length_of_indexer

slc = slice(None, None, -1)
target = [0]

expected = length_of_indexer(slc, target)
actual = len(target[slc])

print(f"length_of_indexer: {expected}")
print(f"actual: {actual}")
print(f"target[slc]: {target[slc]}")
```

Output:
```
length_of_indexer: -1
actual: 1
target[slc]: [0]
```

Additional failing cases:
- `slice(None, None, -1)` on any non-empty list returns negative or incorrect length
- `slice(None, None, -2)` on `[0, 1, 2, 3, 4]` returns incorrect length

## Why This Is A Bug

The `length_of_indexer` function is documented to "Return the expected length of target[indexer]". When given `slice(None, None, -1)` (which reverses the list), the function returns `-1` for a target of length 1, but `[0][::−1]` actually has length 1, not -1.

The bug is in lines 303-316 of `pandas/core/indexers/utils.py`. When handling None values for start/stop with negative steps, the code sets:
- `start = 0` when `start is None`
- `stop = target_len` when `stop is None`

These defaults are correct for positive steps but incorrect for negative steps. For a negative step:
- `start=None` should default to `target_len - 1` (start from the end)
- `stop=None` should default to a value before index 0 (e.g., `-1` after conversion)

## Fix

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -300,14 +300,23 @@ def length_of_indexer(indexer, target=None) -> int:
         start = indexer.start
         stop = indexer.stop
         step = indexer.step
+
+        if step is None:
+            step = 1
+
+        # Handle None values differently for positive vs negative steps
         if start is None:
-            start = 0
+            start = target_len - 1 if step < 0 else 0
         elif start < 0:
             start += target_len
+
         if stop is None or stop > target_len:
-            stop = target_len
+            if step < 0:
+                stop = -1
+            else:
+                stop = target_len
         elif stop < 0:
             stop += target_len
-        if step is None:
-            step = 1
-        elif step < 0:
+
+        if step < 0:
             start, stop = stop + 1, start + 1
             step = -step
```

This fix ensures that when start/stop are None, they are set to the appropriate defaults based on whether the step is positive or negative, before the normalization for negative steps happens.