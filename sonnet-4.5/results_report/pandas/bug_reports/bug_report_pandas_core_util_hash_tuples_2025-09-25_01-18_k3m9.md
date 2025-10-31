# Bug Report: pandas.core.util.hash_tuples Empty List Handling

**Target**: `pandas.core.util.hashing.hash_tuples`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`hash_tuples()` crashes with a `TypeError` when given an empty list, while the related `hash_array()` function handles empty inputs gracefully. This inconsistency violates the principle of least surprise.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.core.util.hashing import hash_tuples
import pytest


def test_hash_tuples_empty():
    hashed = hash_tuples([])
    assert len(hashed) == 0
    assert hashed.dtype == np.uint64
```

**Failing input**: `[]` (empty list)

## Reproducing the Bug

```python
from pandas.core.util.hashing import hash_tuples, hash_array
import numpy as np

empty_arr = np.array([], dtype=np.int64)
hash_arr_result = hash_array(empty_arr)
print(f"hash_array([]) works: {hash_arr_result}")

hash_tuples([])
```

Output:
```
hash_array([]) works: []
Traceback (most recent call last):
  ...
TypeError: Cannot infer number of levels from empty list
```

## Why This Is A Bug

1. **Inconsistency**: `hash_array()` handles empty inputs gracefully by returning an empty uint64 array, but `hash_tuples()` crashes
2. **No documentation**: The docstring doesn't specify that empty lists are invalid
3. **Type compatibility**: The function signature accepts `Iterable[tuple[Hashable, ...]]`, which includes empty iterables
4. **Violates principle of least surprise**: Users would expect consistent behavior across the hashing API

## Fix

```diff
diff --git a/pandas/core/util/hashing.py b/pandas/core/util/hashing.py
index 1234567..abcdefg 100644
--- a/pandas/core/util/hashing.py
+++ b/pandas/core/util/hashing.py
@@ -200,6 +200,10 @@ def hash_tuples(
     """
     if not is_list_like(vals):
         raise TypeError("must be convertible to a list-of-tuples")
+
+    # Handle empty input consistently with hash_array
+    if len(vals) == 0:
+        return np.array([], dtype=np.uint64)

     from pandas import (
         Categorical,
```