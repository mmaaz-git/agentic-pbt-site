# Bug Report: pandas.core.indexers.length_of_indexer Returns Incorrect Length for Empty Arrays

**Target**: `pandas.core.indexers.length_of_indexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `length_of_indexer` function returns incorrect (negative or positive non-zero) lengths when computing the expected length of slice indexing on empty arrays (length 0). This violates the fundamental property that the function should match `len(target[indexer])`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
from pandas.core.indexers import length_of_indexer

@given(
    start=st.one_of(st.none(), st.integers(min_value=-100, max_value=100)),
    stop=st.one_of(st.none(), st.integers(min_value=-100, max_value=100)),
    step=st.one_of(st.none(), st.integers(min_value=-100, max_value=100).filter(lambda x: x != 0)),
    n=st.integers(min_value=0, max_value=100)
)
@settings(max_examples=1000)
def test_length_of_indexer_slice_matches_actual(start, stop, step, n):
    slc = slice(start, stop, step)
    arr = np.arange(n)
    expected_len = len(arr[slc])
    computed_len = length_of_indexer(slc, arr)
    assert expected_len == computed_len
```

**Failing input**: `start=None, stop=-1, step=None, n=0`

## Reproducing the Bug

```python
import numpy as np
from pandas.core.indexers import length_of_indexer

arr = np.arange(0)

print(f"slice(None, -1, None): expected 0, got {length_of_indexer(slice(None, -1, None), arr)}")
print(f"slice(None, -2, None): expected 0, got {length_of_indexer(slice(None, -2, None), arr)}")
print(f"slice(-1, None, None): expected 0, got {length_of_indexer(slice(-1, None, None), arr)}")
print(f"slice(0, -1, None): expected 0, got {length_of_indexer(slice(0, -1, None), arr)}")
```

Output:
```
slice(None, -1, None): expected 0, got -1
slice(None, -2, None): expected 0, got -2
slice(-1, None, None): expected 0, got 1
slice(0, -1, None): expected 0, got -1
```

## Why This Is A Bug

The function's docstring states it should "Return the expected length of target[indexer]". When `target` is an empty array, any slice operation returns an empty array with length 0. However, `length_of_indexer` returns negative or positive non-zero values, violating this contract.

This bug affects code that relies on `length_of_indexer` to pre-allocate arrays or validate operations before performing actual indexing on empty arrays.

## Fix

The issue is in lines 306-310 of `utils.py`:

```python
if start is None:
    start = 0
elif start < 0:
    start += target_len
if stop is None or stop > target_len:
    stop = target_len
elif stop < 0:
    stop += target_len
```

When `target_len = 0` and `stop < 0`, adding `target_len` to `stop` doesn't make it non-negative. The fix is to ensure `stop` is clipped to the valid range `[0, target_len]`:

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -306,9 +306,11 @@ def length_of_indexer(indexer, target=None) -> int:
         elif start < 0:
             start += target_len
         if stop is None or stop > target_len:
             stop = target_len
         elif stop < 0:
             stop += target_len
+            if stop < 0:
+                stop = 0
         if step is None:
             step = 1
         elif step < 0:
```

Similarly, `start` should also be clipped:

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -304,6 +304,8 @@ def length_of_indexer(indexer, target=None) -> int:
             start = 0
         elif start < 0:
             start += target_len
+            if start < 0:
+                start = 0
         if stop is None or stop > target_len:
             stop = target_len
         elif stop < 0:
             stop += target_len
+            if stop < 0:
+                stop = 0
         if step is None:
             step = 1
         elif step < 0:
```