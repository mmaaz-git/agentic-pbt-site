# Bug Report: pandas.core.util.hashing.hash_array Signed Zero Inconsistency

**Target**: `pandas.core.util.hashing.hash_array`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`hash_array()` produces different hash values for `0.0` and `-0.0` even though these values are equal according to IEEE 754 and numpy. This violates the fundamental hash property that equal values must have equal hashes.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
from pandas.core.util.hashing import hash_array

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1))
@settings(max_examples=500)
def test_hash_equal_values_have_equal_hashes(values):
    arr = np.array(values)

    for i in range(len(arr)):
        if arr[i] == 0.0:
            arr_pos = arr.copy()
            arr_pos[i] = 0.0
            arr_neg = arr.copy()
            arr_neg[i] = -0.0

            hash_pos = hash_array(arr_pos)
            hash_neg = hash_array(arr_neg)

            assert np.array_equal(hash_pos, hash_neg), \
                f"Equal arrays should have equal hashes: {arr_pos} vs {arr_neg}"
```

**Failing input**: Arrays containing `-0.0`

## Reproducing the Bug

```python
import numpy as np
from pandas.core.util.hashing import hash_array

arr_pos = np.array([0.0])
arr_neg = np.array([-0.0])

print(f"Arrays equal: {np.array_equal(arr_pos, arr_neg)}")
print(f"hash_array([0.0]):  {hash_array(arr_pos)}")
print(f"hash_array([-0.0]): {hash_array(arr_neg)}")
print(f"Hashes equal: {np.array_equal(hash_array(arr_pos), hash_array(arr_neg))}")
```

Output:
```
Arrays equal: True
hash_array([0.0]):  [0]
hash_array([-0.0]): [2720858781877447050]
Hashes equal: False
```

## Why This Is A Bug

This violates the fundamental hash contract: if `a == b`, then `hash(a)` must equal `hash(b)`.

According to IEEE 754 and numpy:
- `0.0 == -0.0` evaluates to `True`
- `np.array_equal([0.0], [-0.0])` returns `True`
- Python's built-in `hash(0.0) == hash(-0.0)` returns `True`

But pandas' `hash_array([0.0])` and `hash_array([-0.0])` produce different values.

This can cause serious bugs in code that relies on hashing for:
- Groupby operations
- Deduplication
- Hash-based joins
- Index lookups

For example, rows with values `0.0` and `-0.0` in a groupby key would be placed in different groups, even though the values are equal.

## Fix

The issue is in `_hash_ndarray` function at line 305-306 in `hashing.py`:

```python
elif issubclass(dtype.type, np.number) and dtype.itemsize <= 8:
    vals = vals.view(f"u{vals.dtype.itemsize}").astype("u8")
```

This code directly uses the bit representation of floats, which differs for signed zeros. The fix is to normalize signed zeros before hashing:

```diff
--- a/pandas/core/util/hashing.py
+++ b/pandas/core/util/hashing.py
@@ -303,6 +303,9 @@ def _hash_ndarray(
     elif issubclass(dtype.type, (np.datetime64, np.timedelta64)):
         vals = vals.view("i8").astype("u8", copy=False)
     elif issubclass(dtype.type, np.number) and dtype.itemsize <= 8:
+        # Normalize signed zeros to ensure 0.0 and -0.0 hash the same
+        if issubclass(dtype.type, np.floating):
+            vals = vals + 0.0  # Forces -0.0 to become +0.0
         vals = vals.view(f"u{vals.dtype.itemsize}").astype("u8")
     else:
         # object dtypes handled below
```