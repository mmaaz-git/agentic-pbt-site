# Bug Report: RangeIndex._shallow_copy Returns Index for Single-Element Ranges

**Target**: `pandas.core.indexes.range.RangeIndex._shallow_copy`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`RangeIndex._shallow_copy` returns a regular `Index` instead of the memory-efficient `RangeIndex` for single-element equally-spaced arrays, violating its documented optimization goal.

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
        f"Expected RangeIndex for equally-spaced values, got {type(result)}"
```

**Failing input**: `start=0, stop=1, step=1` (single-element range `[0]`)

## Reproducing the Bug

```python
import pandas as pd
from pandas import RangeIndex
import numpy as np

ri = RangeIndex(0, 1, 1)
values = np.array([0])

result = ri._shallow_copy(values)

print(f"Result type: {type(result).__name__}")
print(f"Expected: RangeIndex")
print(f"Actual: {type(result).__name__}")
```

**Output:**
```
Result type: Index
Expected: RangeIndex
Actual: Index
```

## Why This Is A Bug

1. **Violates documented intent**: The code comment at `range.py:473-474` explicitly states:
   > `# GH 46675 & 43885: If values is equally spaced, return a more memory-compact RangeIndex instead of Index with 64-bit dtype`

2. **Single-element arrays are equally spaced**: A single element is trivially equally-spaced (no variance).

3. **Memory inefficiency**: Returns a full `Index` object (stores all values in memory) instead of a compact `RangeIndex` (stores only start/stop/step).

4. **Inconsistent behavior**: Two-element and larger equally-spaced arrays correctly return `RangeIndex`, but single-element arrays do not.

## Root Cause

At `range.py:475-479`, the code uses `unique_deltas(values)` to check if values are equally spaced:

```python
unique_diffs = unique_deltas(values)
if len(unique_diffs) == 1 and unique_diffs[0] != 0:
    diff = unique_diffs[0]
    new_range = range(values[0], values[-1] + diff, diff)
    return type(self)._simple_new(new_range, name=name)
```

For a single-element array `[x]`, `unique_deltas` returns an empty array (no consecutive pairs to compute deltas from). Thus, `len(unique_diffs) == 0`, not `1`, causing the check to fail and falling through to return a regular `Index`.

## Fix

Add special handling for single-element and empty arrays:

```diff
@@ -470,6 +470,11 @@ class RangeIndex(Index):

     if values.dtype.kind == "f":
         return Index(values, name=name, dtype=np.float64)
+
+    # Single-element or empty arrays are trivially equally-spaced
+    if len(values) <= 1:
+        # Return RangeIndex for memory efficiency
+        return type(self)(len(values), name=name) if len(values) == 0 else type(self)._simple_new(range(values[0], values[0] + 1), name=name)
+
     # GH 46675 & 43885: If values is equally spaced, return a
     # more memory-compact RangeIndex instead of Index with 64-bit dtype
     unique_diffs = unique_deltas(values)
```

Alternatively, a cleaner fix:

```diff
@@ -470,6 +470,14 @@ class RangeIndex(Index):

     if values.dtype.kind == "f":
         return Index(values, name=name, dtype=np.float64)
+
+    # Handle edge cases
+    if len(values) == 0:
+        return type(self)(0, name=name)
+    if len(values) == 1:
+        val = values[0]
+        return type(self)._simple_new(range(val, val + 1), name=name)
+
     # GH 46675 & 43885: If values is equally spaced, return a
     # more memory-compact RangeIndex instead of Index with 64-bit dtype
     unique_diffs = unique_deltas(values)
```