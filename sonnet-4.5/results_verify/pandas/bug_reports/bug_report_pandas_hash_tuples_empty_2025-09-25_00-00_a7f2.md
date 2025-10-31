# Bug Report: pandas.core.util.hashing.hash_tuples Empty Input Handling

**Target**: `pandas.core.util.hashing.hash_tuples`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`hash_tuples([])` raises a TypeError instead of handling empty input gracefully like the related functions `hash_array` and `combine_hash_arrays`.

## Property-Based Test

```python
from pandas.core.util.hashing import hash_tuples
import pytest

def test_hash_tuples_empty():
    with pytest.raises(TypeError, match="Cannot infer number of levels from empty list"):
        hash_tuples([])
```

**Failing input**: `[]` (empty list)

## Reproducing the Bug

```python
from pandas.core.util.hashing import hash_tuples, hash_array, combine_hash_arrays
import numpy as np

print("hash_array with empty array:")
result = hash_array(np.array([], dtype=np.int64))
print(f"  Success: {result}")

print("\ncombine_hash_arrays with empty iterator:")
result = combine_hash_arrays(iter([]), 0)
print(f"  Success: {result}")

print("\nhash_tuples with empty list:")
try:
    result = hash_tuples([])
    print(f"  Success: {result}")
except TypeError as e:
    print(f"  Failed: {e}")
```

## Why This Is A Bug

The hashing API is inconsistent. Both `hash_array` and `combine_hash_arrays` gracefully handle empty inputs by returning empty uint64 arrays, but `hash_tuples` raises a TypeError. This violates the principle of least surprise - users would reasonably expect all hashing functions to handle empty collections consistently.

## Fix

```diff
diff --git a/pandas/core/util/hashing.py b/pandas/core/util/hashing.py
index 1234567..abcdefg 100644
--- a/pandas/core/util/hashing.py
+++ b/pandas/core/util/hashing.py
@@ -201,6 +201,10 @@ def hash_tuples(
     if not is_list_like(vals):
         raise TypeError("must be convertible to a list-of-tuples")

+    # Handle empty input gracefully
+    if len(vals) == 0:
+        return np.array([], dtype=np.uint64)
+
     from pandas import (
         Categorical,
         MultiIndex,
```