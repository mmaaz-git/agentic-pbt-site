# Bug Report: pandas.core.indexers.length_of_indexer Incorrect for Range Objects

**Target**: `pandas.core.indexers.length_of_indexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `length_of_indexer` function returns incorrect lengths for range objects when the step size is larger than 1. The function uses floor division `(stop - start) // step` which is incorrect.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from pandas.core.indexers import length_of_indexer

@given(
    start=st.integers(min_value=0, max_value=20),
    stop=st.integers(min_value=0, max_value=20),
    step=st.integers(min_value=1, max_value=5)
)
@settings(max_examples=300)
def test_length_of_indexer_range(start, stop, step):
    indexer = range(start, stop, step)

    computed = length_of_indexer(indexer)
    expected = len(list(indexer))

    assert computed == expected
```

**Failing input**: `range(0, 1, 2)`

## Reproducing the Bug

```python
from pandas.core.indexers import length_of_indexer

test_cases = [
    (0, 1, 2),
    (0, 5, 3),
    (0, 10, 7),
    (5, 10, 1),
    (0, 0, 1),
]

for start, stop, step in test_cases:
    indexer = range(start, stop, step)
    result = length_of_indexer(indexer)
    expected = len(list(indexer))
    match = "✓" if result == expected else "✗"
    print(f"{match} range({start}, {stop}, {step}): computed={result}, expected={expected}")
```

Output:
```
✗ range(0, 1, 2): computed=0, expected=1
✗ range(0, 5, 3): computed=1, expected=2
✗ range(0, 10, 7): computed=1, expected=2
✓ range(5, 10, 1): computed=5, expected=5
✓ range(0, 0, 1): computed=0, expected=0
```

## Why This Is A Bug

The formula `(stop - start) // step` uses floor division, but the correct formula for range length requires ceiling division: `(stop - start + step - 1) // step`. Alternatively, Python's built-in `len()` function handles ranges correctly.

For example:
- `range(0, 1, 2)` contains one element: `[0]`, so length should be 1
- Using the buggy formula: `(1 - 0) // 2 = 0` (incorrect)
- Using the correct formula: `(1 - 0 + 2 - 1) // 2 = 1` (correct)

## Fix

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -325,7 +325,7 @@ def length_of_indexer(indexer, target=None) -> int:
             return indexer.sum()
         return len(indexer)
     elif isinstance(indexer, range):
-        return (indexer.stop - indexer.start) // indexer.step
+        return len(indexer)
     elif not is_list_like_indexer(indexer):
         return 1
```

The fix uses Python's built-in `len()` function for ranges, which correctly handles all cases including edge cases with large step sizes.