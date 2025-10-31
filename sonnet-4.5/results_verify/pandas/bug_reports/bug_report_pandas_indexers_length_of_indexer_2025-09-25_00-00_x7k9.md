# Bug Report: pandas.core.indexers.length_of_indexer Returns Negative Length

**Target**: `pandas.core.indexers.length_of_indexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`length_of_indexer()` returns negative values for empty slices where `start > stop`, violating the fundamental property that length must be non-negative.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import numpy as np
from hypothesis import given, strategies as st, settings
from pandas.core.indexers import length_of_indexer


@settings(max_examples=1000)
@given(st.data())
def test_length_of_indexer_slice_oracle(data):
    target_len = data.draw(st.integers(min_value=0, max_value=1000))
    target = np.arange(target_len)

    start = data.draw(st.one_of(
        st.none(),
        st.integers(min_value=-target_len*2, max_value=target_len*2)
    ))
    stop = data.draw(st.one_of(
        st.none(),
        st.integers(min_value=-target_len*2, max_value=target_len*2)
    ))
    step = data.draw(st.one_of(
        st.none(),
        st.integers(min_value=-100, max_value=100).filter(lambda x: x != 0)
    ))

    indexer = slice(start, stop, step)

    computed_length = length_of_indexer(indexer, target)
    actual_length = len(target[indexer])

    assert computed_length == actual_length
```

**Failing input**: `slice(1, 0, None)` on array of length 1

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import numpy as np
from pandas.core.indexers import length_of_indexer

target = np.array([0])
indexer = slice(1, 0, None)

computed = length_of_indexer(indexer, target)
actual = len(target[indexer])

print(f"Computed: {computed}")
print(f"Actual: {actual}")
print(f"Bug: {computed} != {actual}")
```

Output:
```
Computed: -1
Actual: 0
Bug: -1 != 0
```

## Why This Is A Bug

The function's docstring states it should "Return the expected length of target[indexer]". Lengths cannot be negative. When `start > stop` with a positive step, the slice is empty and should have length 0, not -1.

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