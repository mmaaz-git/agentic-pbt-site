# Bug Report: pandas.core.indexers.length_of_indexer Negative Return Value

**Target**: `pandas.core.indexers.length_of_indexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `length_of_indexer` function returns negative values when given a slice with `start > len(target)`, instead of returning 0 as expected.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.core.indexers import length_of_indexer

@settings(max_examples=1000)
@given(
    target_len=st.integers(min_value=0, max_value=200),
    start=st.integers(min_value=-100, max_value=200) | st.none(),
    stop=st.integers(min_value=-100, max_value=200) | st.none(),
    step=st.integers(min_value=-20, max_value=20).filter(lambda x: x != 0) | st.none(),
)
def test_length_of_indexer_slice_comprehensive(target_len, start, stop, step):
    target = list(range(target_len))
    slc = slice(start, stop, step)

    computed_length = length_of_indexer(slc, target)
    actual_length = len(target[slc])

    assert computed_length == actual_length, \
        f"For slice({start}, {stop}, {step}) on len={target_len}: " \
        f"length_of_indexer={computed_length}, actual={actual_length}"
```

**Failing input**: `target_len=0, start=1, stop=None, step=None`

## Reproducing the Bug

```python
from pandas.core.indexers import length_of_indexer

target = []
slc = slice(1, None, None)

computed = length_of_indexer(slc, target)
actual = len(target[slc])

print(f"len(target[slc]) = {actual}")
print(f"length_of_indexer(slc, target) = {computed}")
```

Output:
```
len(target[slc]) = 0
length_of_indexer(slc, target) = -1
```

## Why This Is A Bug

The function is documented to "Return the expected length of target[indexer]". When `start` is beyond the array bounds, Python's slice behavior returns an empty sequence (length 0), but `length_of_indexer` returns a negative number. This violates the fundamental property that a length should never be negative.

Additional failing cases:
- `slice(10, None)` on list of length 5: returns -5 instead of 0
- `slice(1, None)` on empty list: returns -1 instead of 0

## Fix

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -305,6 +305,8 @@ def length_of_indexer(indexer, target=None) -> int:
             start = 0
         elif start < 0:
             start += target_len
+        start = max(0, min(start, target_len))
+
         if stop is None or stop > target_len:
             stop = target_len
         elif stop < 0:
```