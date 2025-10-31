# Bug Report: pandas.core.indexers.length_of_indexer Range Length Calculation

**Target**: `pandas.core.indexers.length_of_indexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `length_of_indexer` function returns incorrect lengths for `range` objects when the step size is larger than the distance between start and stop, violating the fundamental property that `length_of_indexer(indexer) == len(indexer)`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from pandas.core.indexers import length_of_indexer

@given(
    start=st.integers(min_value=0, max_value=50),
    stop=st.integers(min_value=0, max_value=100),
    step=st.integers(min_value=1, max_value=10)
)
def test_length_of_indexer_range_consistency(start, stop, step):
    assume(start < stop)

    r = range(start, stop, step)
    expected_length = len(r)
    predicted_length = length_of_indexer(r)

    assert expected_length == predicted_length
```

**Failing input**: `range(0, 1, 2)`

## Reproducing the Bug

```python
from pandas.core.indexers import length_of_indexer

r = range(0, 1, 2)
expected = len(r)
predicted = length_of_indexer(r)

print(f"Expected: {expected}")
print(f"Predicted: {predicted}")
print(f"Bug: {expected != predicted}")
```

Output:
```
Expected: 1
Predicted: 0
Bug: True
```

## Why This Is A Bug

The function's docstring states "Return the expected length of target[indexer]", but it returns an incorrect result for range objects when the step is larger than the range span. This violates the core contract of the function.

The bug occurs in `utils.py` line 326:

```python
elif isinstance(indexer, range):
    return (indexer.stop - indexer.start) // indexer.step
```

For `range(0, 1, 2)`:
- Correct: Python's `len(range(0, 1, 2))` = 1
- Buggy formula: `(1 - 0) // 2` = 0

The formula fails to account for the ceiling division needed when `step` doesn't evenly divide `(stop - start)`.

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