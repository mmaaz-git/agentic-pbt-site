# Bug Report: length_of_indexer Negative Step Calculation

**Target**: `pandas.core.indexers.utils.length_of_indexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`length_of_indexer` returns incorrect (negative) length when given a slice with a negative step and `None` stop value on certain target lengths.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import numpy as np
from hypothesis import given, strategies as st, assume, settings
import pandas.core.indexers as indexers


@given(
    slc=st.builds(
        slice,
        st.one_of(st.none(), st.integers(min_value=-100, max_value=100)),
        st.one_of(st.none(), st.integers(min_value=-100, max_value=100)),
        st.one_of(st.none(), st.integers(min_value=-10, max_value=10).filter(lambda x: x != 0))
    ),
    target_len=st.integers(min_value=0, max_value=100)
)
@settings(max_examples=1000)
def test_length_of_indexer_slice_matches_actual(slc, target_len):
    target = list(range(target_len))
    calculated_length = indexers.length_of_indexer(slc, target)
    actual_sliced = target[slc]
    actual_length = len(actual_sliced)
    assert calculated_length == actual_length
```

**Failing input**: `slc=slice(None, None, -1), target_len=1`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas.core.indexers as indexers

slc = slice(None, None, -1)
target = [0]

calculated_length = indexers.length_of_indexer(slc, target)
actual_length = len(target[slc])

print(f"Calculated length: {calculated_length}")
print(f"Actual length: {actual_length}")
```

Output:
```
Calculated length: -1
Actual length: 1
```

## Why This Is A Bug

The function `length_of_indexer` is documented to "Return the expected length of target[indexer]". When given `slice(None, None, -1)` and a target of length 1, the actual result of `target[slice(None, None, -1)]` is `[0]` with length 1, but the function returns -1.

This violates the metamorphic property that the calculated length should match the actual length of the sliced result.

## Fix

The bug occurs in the handling of negative steps when `stop is None`. In Python, when slicing with a negative step and `stop=None`, the slice goes from start to the beginning of the sequence. The current code incorrectly sets `stop = target_len` before swapping start/stop for negative steps.

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -304,7 +304,11 @@ def length_of_indexer(indexer, target=None) -> int:
             start = 0
         elif start < 0:
             start += target_len
-        if stop is None or stop > target_len:
+        if stop is None:
+            if step is not None and step < 0:
+                stop = -1
+            else:
+                stop = target_len
+        elif stop > target_len:
             stop = target_len
         elif stop < 0:
             stop += target_len
```