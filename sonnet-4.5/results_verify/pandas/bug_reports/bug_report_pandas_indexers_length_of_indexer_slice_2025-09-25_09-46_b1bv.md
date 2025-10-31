# Bug Report: pandas.core.indexers.length_of_indexer Negative Length for Empty Slices

**Target**: `pandas.core.indexers.length_of_indexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`length_of_indexer` returns negative lengths for slices with negative stop indices on empty targets, when it should return 0 to match Python's actual slicing behavior.

## Property-Based Test

```python
from hypothesis import assume, given, settings, strategies as st
import pandas.core.indexers as indexers

@given(
    start=st.one_of(st.integers(), st.none()),
    stop=st.one_of(st.integers(), st.none()),
    step=st.one_of(st.integers(), st.none()),
    n=st.integers(min_value=0, max_value=100)
)
@settings(max_examples=500)
def test_length_of_indexer_matches_actual_slicing(start, stop, step, n):
    assume(step != 0)
    slc = slice(start, stop, step)
    target = list(range(n))

    try:
        expected_len = len(target[slc])
        calculated_len = indexers.length_of_indexer(slc, target)
        assert calculated_len == expected_len
    except (ValueError, ZeroDivisionError):
        pass
```

**Failing input**: `start=None, stop=-1, step=None, n=0`

## Reproducing the Bug

```python
import pandas.core.indexers as indexers

target = []
slc = slice(None, -1, None)

print(f"len(target[{slc}]) = {len(target[slc])}")
print(f"length_of_indexer({slc}, {target}) = {indexers.length_of_indexer(slc, target)}")
```

Output:
```
len(target[slice(None, -1, None)]) = 0
length_of_indexer(slice(None, -1, None), []) = -1
```

## Why This Is A Bug

The function `length_of_indexer` is documented to "Return the expected length of target[indexer]". When slicing an empty list with a negative stop index like `[][:−1]`, Python returns an empty list with length 0. However, `length_of_indexer` returns -1, violating its documented contract.

The bug occurs because when `stop < 0`, the code does `stop += target_len`, which equals `-1 + 0 = -1` for empty targets. The formula `(stop - start + step - 1) // step` then evaluates to `(-1 - 0 + 1 - 1) // 1 = -1`.

## Fix

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -20,7 +20,7 @@ def length_of_indexer(indexer, target=None) -> int:
         if stop is None or stop > target_len:
             stop = target_len
         elif stop < 0:
             stop += target_len
+            stop = max(0, stop)
         if step is None:
             step = 1
         elif step < 0:
             start, stop = stop + 1, start + 1
             step = -step
-        return (stop - start + step - 1) // step
+        return max(0, (stop - start + step - 1) // step)
```