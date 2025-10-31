# Bug Report: pandas.core.indexers.length_of_indexer Returns Negative Length for Negative Step Slices

**Target**: `pandas.core.indexers.length_of_indexer`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `length_of_indexer` function returns negative lengths when computing the length of slices with negative steps, violating the fundamental invariant that lengths must be non-negative integers.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.core.indexers import length_of_indexer

@given(
    start=st.one_of(st.none(), st.integers(min_value=-100, max_value=100)),
    stop=st.one_of(st.none(), st.integers(min_value=-100, max_value=100)),
    step=st.one_of(st.none(), st.integers(min_value=-100, max_value=100).filter(lambda x: x != 0)),
    target_len=st.integers(min_value=0, max_value=100)
)
@settings(max_examples=500)
def test_length_of_indexer_slice_correctness(start, stop, step, target_len):
    slc = slice(start, stop, step)
    target = list(range(target_len))

    computed_len = length_of_indexer(slc, target)
    actual_len = len(target[slc])

    assert computed_len == actual_len, \
        f"length_of_indexer({slc}, len={target_len}) = {computed_len}, but actual length = {actual_len}"
```

**Failing input**: `slice(None, None, -1)` with `target_len=1`

## Reproducing the Bug

```python
from pandas.core.indexers import length_of_indexer

target = [0]
slc = slice(None, None, -1)

computed_len = length_of_indexer(slc, target)
actual_len = len(target[slc])

print(f"length_of_indexer result: {computed_len}")
print(f"Actual length: {actual_len}")
print(f"Target[slc]: {target[slc]}")
```

**Output:**
```
length_of_indexer result: -1
Actual length: 1
Target[slc]: [0]
```

**Additional failing examples:**
- `slice(None, None, -1)` on list of length 5: returns -5, should return 5
- `slice(None, None, -2)` on list of length 4: returns -2, should return 2
- `slice(5, None, -1)` on list of length 7: returns -2, should return 6
- `slice(None, 2, -1)` on list of length 6: returns -2, should return 3

## Why This Is A Bug

The function is documented to "Return the expected length of target[indexer]" and has return type `int`, which should always be non-negative for valid slices. Returning negative lengths violates this contract and will cause errors in any code that uses this function to allocate arrays, validate indices, or perform other length-dependent operations.

## Root Cause

In `pandas/core/indexers/utils.py` at lines 278-295, the function manually implements slice normalization logic for negative steps. This implementation is buggy and doesn't handle all edge cases correctly, particularly:

1. When both start and stop are `None` with a negative step
2. When only start or only stop is specified with a negative step

The manual normalization incorrectly swaps start/stop and then applies a formula that produces negative results.

## Fix

The simplest and most correct fix is to use Python's built-in `slice.indices()` method, which correctly handles all edge cases including negative steps:

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -275,23 +275,7 @@ def length_of_indexer(indexer, target=None) -> int:
     """
     if target is not None and isinstance(indexer, slice):
-        target_len = len(target)
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
+        return len(range(*indexer.indices(len(target))))
     elif isinstance(indexer, (ABCSeries, ABCIndex, np.ndarray, list)):
```

This fix replaces 18 lines of buggy manual logic with a single line that delegates to Python's correct implementation.