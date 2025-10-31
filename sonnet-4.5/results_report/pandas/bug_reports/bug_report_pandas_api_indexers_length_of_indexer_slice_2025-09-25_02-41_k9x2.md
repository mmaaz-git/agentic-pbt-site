# Bug Report: pandas.api.indexers length_of_indexer Empty Slice

**Target**: `pandas.core.indexers.utils.length_of_indexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `length_of_indexer()` function incorrectly returns -1 for empty slices (where start > stop), but should return 0 to match Python's slice behavior.

## Property-Based Test

```python
@given(
    st.integers(min_value=-20, max_value=20),
    st.integers(min_value=-20, max_value=20),
    st.one_of(st.integers(min_value=-5, max_value=5).filter(lambda x: x != 0), st.none())
)
@settings(max_examples=500)
def test_length_of_indexer_slice_property(start, stop, step):
    target = list(range(50))
    slc = slice(start, stop, step)

    computed_length = length_of_indexer(slc, target)
    actual_length = len(target[slc])

    assert computed_length == actual_length
```

**Failing input**: `slice(1, 0, None)`

## Reproducing the Bug

```python
from pandas.core.indexers.utils import length_of_indexer

target = list(range(50))
slc = slice(1, 0, None)

computed = length_of_indexer(slc, target)
actual = len(target[slc])

print(f"length_of_indexer(slice(1, 0, None), target) = {computed}")
print(f"len(target[slice(1, 0, None)]) = {actual}")
assert computed == actual
```

## Why This Is A Bug

The function's purpose is to "Return the expected length of target[indexer]". For empty slices where start >= stop (with positive step), Python returns an empty sequence with length 0. The function returns -1, which is incorrect and could cause issues in downstream code expecting non-negative lengths.

The bug is in line 316 of `/pandas/core/indexers/utils.py`:
```python
return (stop - start + step - 1) // step
```

For slice(1, 0, None) with step=1: `(0 - 1 + 1 - 1) // 1 = -1`

## Fix

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -313,7 +313,7 @@ def length_of_indexer(indexer, target=None) -> int:
         elif step < 0:
             start, stop = stop + 1, start + 1
             step = -step
-        return (stop - start + step - 1) // step
+        return max(0, (stop - start + step - 1) // step)
     elif isinstance(indexer, (ABCSeries, ABCIndex, np.ndarray, list)):
         if isinstance(indexer, list):
             indexer = np.array(indexer)
```