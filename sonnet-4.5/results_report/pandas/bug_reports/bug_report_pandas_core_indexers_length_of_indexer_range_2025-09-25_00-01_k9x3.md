# Bug Report: pandas.core.indexers.length_of_indexer Range Empty Check

**Target**: `pandas.core.indexers.length_of_indexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `length_of_indexer` function returns negative lengths for empty ranges (where start > stop), violating the fundamental property that `length_of_indexer(indexer, target) == len(indexer)`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.core.indexers import length_of_indexer

@given(
    start=st.integers(min_value=0, max_value=100),
    stop=st.integers(min_value=0, max_value=100),
    step=st.integers(min_value=1, max_value=10)
)
def test_length_of_indexer_range_consistency(start, stop, step):
    rng = range(start, stop, step)
    expected_length = len(rng)
    predicted_length = length_of_indexer(rng)

    assert expected_length == predicted_length
```

**Failing input**: `range(1, 0, 1)`

## Reproducing the Bug

```python
from pandas.core.indexers import length_of_indexer

rng = range(1, 0, 1)

expected_length = len(rng)
predicted_length = length_of_indexer(rng)

print(f"Expected length: {expected_length}")
print(f"Predicted length: {predicted_length}")
```

## Why This Is A Bug

The function's docstring states "Return the expected length of target[indexer]", but it returns -1 instead of 0 for empty ranges where start > stop. Python's `range(1, 0, 1)` is empty and has length 0, not -1.

The bug occurs in lines 325-326 of `utils.py`:

```python
elif isinstance(indexer, range):
    return (indexer.stop - indexer.start) // indexer.step
```

For `range(1, 0, 1)`:
- `(0 - 1) // 1 = -1`

The formula doesn't account for empty ranges where `start > stop` (for positive steps) or `start < stop` (for negative steps).

## Fix

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -324,7 +324,12 @@ def length_of_indexer(indexer, target=None) -> int:
         return len(indexer)
     elif isinstance(indexer, range):
-        return (indexer.stop - indexer.start) // indexer.step
+        if indexer.step > 0:
+            length = max(0, (indexer.stop - indexer.start + indexer.step - 1) // indexer.step)
+        else:
+            length = max(0, (indexer.start - indexer.stop - indexer.step - 1) // (-indexer.step))
+        return length
     elif not is_list_like_indexer(indexer):
         return 1
     raise AssertionError("cannot find the length of the indexer")
```