# Bug Report: pandas.core.indexers.length_of_indexer Negative Length for Empty Ranges

**Target**: `pandas.core.indexers.length_of_indexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`length_of_indexer` returns negative lengths for empty `range` objects (where `stop <= start` with positive step), when it should return 0 to match Python's `len(range(...))` behavior.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
import pandas.core.indexers as indexers

@given(
    start=st.integers(min_value=0, max_value=100),
    stop=st.integers(min_value=0, max_value=100),
    step=st.integers(min_value=1, max_value=10)
)
@settings(max_examples=500)
def test_length_of_indexer_range(start, stop, step):
    r = range(start, stop, step)
    result = indexers.length_of_indexer(r)
    expected = len(r)
    assert result == expected
```

**Failing input**: `start=1, stop=0, step=1`

## Reproducing the Bug

```python
import pandas.core.indexers as indexers

r = range(1, 0, 1)
print(f"len(range(1, 0, 1)) = {len(r)}")
print(f"length_of_indexer(range(1, 0, 1)) = {indexers.length_of_indexer(r)}")

r = range(10, 0, 2)
print(f"len(range(10, 0, 2)) = {len(r)}")
print(f"length_of_indexer(range(10, 0, 2)) = {indexers.length_of_indexer(r)}")
```

Output:
```
len(range(1, 0, 1)) = 0
length_of_indexer(range(1, 0, 1)) = -1
len(range(10, 0, 2)) = 0
length_of_indexer(range(10, 0, 2)) = -5
```

## Why This Is A Bug

The function `length_of_indexer` is documented to "Return the expected length of target[indexer]". For `range` objects, it should return the same value as Python's built-in `len()`. When `stop <= start` with a positive step, `range` is empty and has length 0, but `length_of_indexer` returns a negative number.

The bug occurs because the formula `(indexer.stop - indexer.start) // indexer.step` doesn't handle the case where `stop < start` (for positive step), producing negative results instead of 0.

## Fix

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -32,7 +32,7 @@ def length_of_indexer(indexer, target=None) -> int:
             return indexer.sum()
         return len(indexer)
     elif isinstance(indexer, range):
-        return (indexer.stop - indexer.start) // indexer.step
+        return max(0, (indexer.stop - indexer.start + indexer.step - 1) // indexer.step) if indexer.step > 0 else max(0, (indexer.start - indexer.stop - indexer.step - 1) // (-indexer.step))
     elif not is_list_like_indexer(indexer):
         return 1
```

Alternatively, a simpler fix:
```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -32,7 +32,7 @@ def length_of_indexer(indexer, target=None) -> int:
             return indexer.sum()
         return len(indexer)
     elif isinstance(indexer, range):
-        return (indexer.stop - indexer.start) // indexer.step
+        return len(indexer)
```