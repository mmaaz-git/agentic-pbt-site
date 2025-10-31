# Bug Report: pandas.core.indexers.length_of_indexer Incorrect Length for Negative Step Slices

**Target**: `pandas.core.indexers.length_of_indexer`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `length_of_indexer` function returns incorrect (negative) lengths for slices with negative steps when `start` and/or `stop` are `None`. For example, `slice(None, None, -1)` on a list of length 5 returns -5 instead of 5.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas.core.indexers as indexers

@given(
    start=st.one_of(st.none(), st.integers(min_value=-50, max_value=50)),
    stop=st.one_of(st.none(), st.integers(min_value=-50, max_value=50)),
    step=st.integers(min_value=-20, max_value=20).filter(lambda x: x != 0),
    n=st.integers(min_value=1, max_value=50)
)
def test_length_of_indexer_slice_comprehensive(start, stop, step, n):
    slc = slice(start, stop, step)
    target = list(range(n))

    computed_length = indexers.length_of_indexer(slc, target)
    actual_length = len(target[slc])

    assert computed_length == actual_length, \
        f"slice({start}, {stop}, {step}) on list of length {n}: " \
        f"length_of_indexer returned {computed_length}, actual length is {actual_length}"
```

**Failing input**: `slice(None, None, -1)` with `n=5`

## Reproducing the Bug

```python
import pandas.core.indexers as indexers

target = [0, 1, 2, 3, 4]
slc = slice(None, None, -1)

computed = indexers.length_of_indexer(slc, target)
actual = len(target[slc])

print(f"Computed length: {computed}")
print(f"Actual length: {actual}")
print(f"Actual result: {target[slc]}")
```

Output:
```
Computed length: -5
Actual length: 5
Actual result: [4, 3, 2, 1, 0]
```

## Why This Is A Bug

When a slice has a negative step and `None` for start/stop, Python uses different default values than for positive steps:
- For `slice(None, None, -1)` on a sequence of length n:
  - `start` should default to `n-1` (last element)
  - `stop` should default to `-1` (before first element)
- But `length_of_indexer` incorrectly uses `start=0` and `stop=n` as defaults regardless of step sign.

This causes the swap logic (`start, stop = stop + 1, start + 1` when `step < 0`) to produce incorrect values, resulting in negative lengths.

The correct approach is to use Python's `slice.indices(length)` method which handles all these cases correctly.

## Fix

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -318,22 +318,7 @@ def length_of_indexer(indexer, target=None) -> int:
     """
     if target is not None and isinstance(indexer, slice):
         target_len = len(target)
-        start = indexer.start
-        stop = indexer.stop
-        step = indexer.step
-        if start is None:
-            start = 0
-        elif start < 0:
-            start += target_len
-        if stop is None or stop > target_len:
-            stop = target_len
-        elif stop < 0:
-            stop += target_len
-        if step is None:
-            step = 1
-        elif step < 0:
-            start, stop = stop + 1, start + 1
-            step = -step
-        return (stop - start + step - 1) // step
+        start, stop, step = indexer.indices(target_len)
+        return len(range(start, stop, step))
     elif isinstance(indexer, (ABCSeries, ABCIndex, np.ndarray, list)):
         if isinstance(indexer, list):
```

This fix uses Python's built-in `slice.indices()` method which correctly normalizes start, stop, and step values, then uses `len(range(start, stop, step))` to compute the length, which handles all cases correctly including negative steps and edge cases.