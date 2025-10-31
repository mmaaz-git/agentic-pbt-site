# Bug Report: pandas.api.indexers.length_of_indexer Negative Length Return

**Target**: `pandas.core.indexers.utils.length_of_indexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`length_of_indexer` returns negative values for slices with negative stop indices when the target is empty, violating its contract to return the expected length of `target[indexer]`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.core.indexers.utils import length_of_indexer

@given(
    indexer=st.slices(20),
    target_len=st.integers(min_value=0, max_value=20),
)
def test_length_of_indexer_slice_matches_actual(indexer, target_len):
    target = list(range(target_len))

    expected_len = length_of_indexer(indexer, target)
    actual_result = target[indexer]
    actual_len = len(actual_result)

    assert expected_len == actual_len, (
        f"length_of_indexer({indexer}, len={target_len}) = {expected_len}, "
        f"but len(target[indexer]) = {actual_len}"
    )
```

**Failing input**: `indexer=slice(0, -20, None)`, `target_len=0`

## Reproducing the Bug

```python
from pandas.core.indexers.utils import length_of_indexer

target = []
indexer = slice(0, -20, None)

result = length_of_indexer(indexer, target)
actual_length = len(target[indexer])

print(f"length_of_indexer returned: {result}")
print(f"Actual length: {actual_length}")
assert result == actual_length
```

Output:
```
length_of_indexer returned: -20
Actual length: 0
AssertionError
```

## Why This Is A Bug

The function's docstring states it should "Return the expected length of target[indexer]". Lengths cannot be negative. When slicing an empty list with `slice(0, -20)`, Python returns an empty list (length 0), but `length_of_indexer` incorrectly returns -20.

The issue occurs because when `target_len = 0` and `stop = -20`:
1. The code does `stop += target_len` → `stop = -20 + 0 = -20`
2. It never clamps negative results to 0
3. Returns `(-20 - 0 + 1 - 1) // 1 = -20`

## Fix

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -308,6 +308,8 @@ def length_of_indexer(indexer, target=None) -> int:
             stop = target_len
         elif stop < 0:
             stop += target_len
+            if stop < 0:
+                stop = 0
         if step is None:
             step = 1
         elif step < 0:
```

Alternatively, the final calculation could use `max(0, (stop - start + step - 1) // step)` to ensure non-negative results.