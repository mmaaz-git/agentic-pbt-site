# Bug Report: pandas.core.indexes.range.RangeIndex._shallow_copy Returns Index Instead of RangeIndex for Single-Element Arrays

**Target**: `pandas.core.indexes.range.RangeIndex._shallow_copy`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `RangeIndex._shallow_copy` method incorrectly returns a regular `Index` object instead of the memory-efficient `RangeIndex` when copying single-element equally-spaced arrays, violating its documented optimization goal to return RangeIndex for all equally-spaced values.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd
from pandas import RangeIndex
import numpy as np

@given(
    st.integers(min_value=-1000, max_value=1000),
    st.integers(min_value=-1000, max_value=1000),
    st.integers(min_value=-100, max_value=100).filter(lambda x: x != 0)
)
def test_rangeindex_shallow_copy_with_equally_spaced_values(start, stop, step):
    """RangeIndex._shallow_copy should return RangeIndex for equally spaced values."""
    ri = RangeIndex(start, stop, step)
    if len(ri) == 0:
        return

    values = np.array(list(ri))
    result = ri._shallow_copy(values)

    # BUG: Fails for single-element ranges
    assert isinstance(result, RangeIndex), \
        f"Expected RangeIndex for equally-spaced values, got {type(result).__name__} for values={values}"

# Run the test
if __name__ == "__main__":
    test_rangeindex_shallow_copy_with_equally_spaced_values()
```

<details>

<summary>
**Failing input**: `start=0, stop=1, step=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 26, in <module>
    test_rangeindex_shallow_copy_with_equally_spaced_values()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 7, in test_rangeindex_shallow_copy_with_equally_spaced_values
    st.integers(min_value=-1000, max_value=1000),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 21, in test_rangeindex_shallow_copy_with_equally_spaced_values
    assert isinstance(result, RangeIndex), \
           ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected RangeIndex for equally-spaced values, got Index for values=[0]
Falsifying example: test_rangeindex_shallow_copy_with_equally_spaced_values(
    start=0,
    stop=1,
    step=1,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/indexes/base.py:667
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/indexes/range.py:481
```
</details>

## Reproducing the Bug

```python
import pandas as pd
from pandas import RangeIndex
import numpy as np

# Create a single-element RangeIndex
ri = RangeIndex(0, 1, 1)
print(f"Original RangeIndex: {ri}")
print(f"Original RangeIndex values: {list(ri)}")
print(f"Original type: {type(ri).__name__}")
print()

# Create a numpy array with the single element
values = np.array([0])
print(f"Values to shallow copy: {values}")
print()

# Call _shallow_copy
result = ri._shallow_copy(values)

print(f"Result after _shallow_copy: {result}")
print(f"Result type: {type(result).__name__}")
print(f"Expected type: RangeIndex (for memory efficiency with equally-spaced values)")
print()

# Check if it's a RangeIndex
is_range_index = isinstance(result, RangeIndex)
print(f"Is result a RangeIndex? {is_range_index}")
print()

# Compare with multi-element case
print("=" * 50)
print("Comparison with 2-element RangeIndex:")
ri2 = RangeIndex(0, 2, 1)
values2 = np.array([0, 1])
result2 = ri2._shallow_copy(values2)
print(f"2-element values: {values2}")
print(f"2-element result type: {type(result2).__name__}")
print(f"Is 2-element result a RangeIndex? {isinstance(result2, RangeIndex)}")
```

<details>

<summary>
Single-element array returns Index instead of RangeIndex
</summary>
```
Original RangeIndex: RangeIndex(start=0, stop=1, step=1)
Original RangeIndex values: [0]
Original type: RangeIndex

Values to shallow copy: [0]

Result after _shallow_copy: Index([0], dtype='int64')
Result type: Index
Expected type: RangeIndex (for memory efficiency with equally-spaced values)

Is result a RangeIndex? False

==================================================
Comparison with 2-element RangeIndex:
2-element values: [0 1]
2-element result type: RangeIndex
Is 2-element result a RangeIndex? True
```
</details>

## Why This Is A Bug

The implementation violates its documented intent and exhibits inconsistent behavior. The code comment at lines 473-474 of `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/indexes/range.py` explicitly states:

> "GH 46675 & 43885: If values is equally spaced, return a more memory-compact RangeIndex instead of Index with 64-bit dtype"

A single-element array is mathematically equally-spaced (trivially so, with no variance between elements since there's only one element). The current implementation fails because:

1. The `unique_deltas` function returns an empty array for single-element inputs (no consecutive pairs to compute deltas from)
2. The check `len(unique_diffs) == 1 and unique_diffs[0] != 0` fails when `unique_diffs` is empty
3. The code falls through to return a regular `Index` object instead of the memory-efficient `RangeIndex`

This results in inconsistent behavior where two-element and larger equally-spaced arrays correctly return `RangeIndex`, but single-element arrays do not. This wastes memory by storing the actual values in an Index object rather than just storing the start/stop/step parameters of a RangeIndex.

## Relevant Context

The bug occurs in the `_shallow_copy` method which is an internal method used when pandas needs to create a copy of an index with different values. While this is not a public API method that users would call directly, it can affect memory usage in pandas operations that internally use this method.

The referenced GitHub issues (GH 46675 and GH 43885) indicate that this memory optimization was intentionally added to pandas to improve memory efficiency for equally-spaced integer sequences. The optimization works correctly for sequences with 2 or more elements but fails for the edge case of single-element sequences.

Relevant code location: `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/indexes/range.py:467-481`

## Proposed Fix

```diff
--- a/pandas/core/indexes/range.py
+++ b/pandas/core/indexes/range.py
@@ -470,6 +470,13 @@ class RangeIndex(Index):

         if values.dtype.kind == "f":
             return Index(values, name=name, dtype=np.float64)
+
+        # Handle single-element arrays which are trivially equally-spaced
+        if len(values) == 1:
+            val = values[0]
+            return type(self)._simple_new(range(val, val + 1), name=name)
+        elif len(values) == 0:
+            return type(self)._simple_new(range(0), name=name)
+
         # GH 46675 & 43885: If values is equally spaced, return a
         # more memory-compact RangeIndex instead of Index with 64-bit dtype
         unique_diffs = unique_deltas(values)
```