# Bug Report: pandas.core.indexers length_of_indexer Incorrect for range Objects

**Target**: `pandas.core.indexers.length_of_indexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`length_of_indexer` returns incorrect length for `range` objects when the step doesn't evenly divide the range.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from pandas.core.indexers import length_of_indexer

@given(
    start=st.integers(min_value=0, max_value=100),
    stop=st.integers(min_value=0, max_value=100),
    step=st.integers(min_value=1, max_value=10),
)
def test_length_of_indexer_range(start, stop, step):
    assume(start < stop)
    indexer = range(start, stop, step)
    expected_len = length_of_indexer(indexer)
    actual_len = len(indexer)
    assert expected_len == actual_len
```

**Failing input**: `range(0, 1, 2)`

## Reproducing the Bug

```python
from pandas.core.indexers import length_of_indexer

r = range(0, 1, 2)
print(f"range(0, 1, 2) = {list(r)}")
print(f"length_of_indexer: {length_of_indexer(r)}")
print(f"Actual length: {len(r)}")

r2 = range(0, 10, 3)
print(f"\nrange(0, 10, 3) = {list(r2)}")
print(f"length_of_indexer: {length_of_indexer(r2)}")
print(f"Actual length: {len(r2)}")
```

Output:
```
range(0, 1, 2) = [0]
length_of_indexer: 0
Actual length: 1

range(0, 10, 3) = [0, 3, 6, 9]
length_of_indexer: 3
Actual length: 4
```

## Why This Is A Bug

The formula `(indexer.stop - indexer.start) // indexer.step` on line 326 doesn't correctly calculate range length when the step doesn't evenly divide the range. The correct calculation needs to round up, not down.

## Fix

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -323,7 +323,7 @@ def length_of_indexer(indexer, target=None) -> int:
             return indexer.sum()
         return len(indexer)
     elif isinstance(indexer, range):
-        return (indexer.stop - indexer.start) // indexer.step
+        return len(indexer)
     elif not is_list_like_indexer(indexer):
         return 1
     raise AssertionError("cannot find the length of the indexer")
```