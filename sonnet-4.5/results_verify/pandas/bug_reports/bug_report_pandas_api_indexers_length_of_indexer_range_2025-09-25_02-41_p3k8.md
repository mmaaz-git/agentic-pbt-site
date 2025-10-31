# Bug Report: pandas.api.indexers length_of_indexer Range Length Calculation

**Target**: `pandas.core.indexers.utils.length_of_indexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `length_of_indexer()` function uses incorrect formula for calculating range lengths, causing it to return 0 for ranges like `range(0, 1, 2)` which actually have length 1.

## Property-Based Test

```python
@given(
    st.integers(min_value=0, max_value=1000),
    st.integers(min_value=0, max_value=100),
    st.integers(min_value=1, max_value=20)
)
@settings(max_examples=300)
def test_range_length_of_indexer_property(start, stop, step):
    assume(start < stop)
    indexer = range(start, stop, step)

    computed_length = length_of_indexer(indexer)
    expected_length = len(indexer)

    assert computed_length == expected_length
```

**Failing input**: `range(0, 1, 2)`

## Reproducing the Bug

```python
from pandas.core.indexers.utils import length_of_indexer

indexer = range(0, 1, 2)
computed = length_of_indexer(indexer)
actual = len(indexer)

print(f"length_of_indexer(range(0, 1, 2)) = {computed}")
print(f"len(range(0, 1, 2)) = {actual}")
assert computed == actual
```

Output:
```
length_of_indexer(range(0, 1, 2)) = 0
len(range(0, 1, 2)) = 1
AssertionError
```

## Why This Is A Bug

The function uses floor division `(stop - start) // step` which is incorrect for range length calculation. The correct formula requires ceiling division to handle cases where the step doesn't evenly divide the range. For `range(0, 1, 2)`: there's one element (0), but `(1-0)//2 = 0`.

This affects rolling window operations and other pandas functionality that relies on accurate indexer length calculations.

## Fix

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -323,7 +323,9 @@ def length_of_indexer(indexer, target=None) -> int:
             return indexer.sum()
         return len(indexer)
     elif isinstance(indexer, range):
-        return (indexer.stop - indexer.start) // indexer.step
+        if indexer.step > 0:
+            return max(0, (indexer.stop - indexer.start + indexer.step - 1) // indexer.step)
+        return max(0, (indexer.start - indexer.stop - indexer.step - 1) // (-indexer.step))
     elif not is_list_like_indexer(indexer):
         return 1
     raise AssertionError("cannot find the length of the indexer")
```