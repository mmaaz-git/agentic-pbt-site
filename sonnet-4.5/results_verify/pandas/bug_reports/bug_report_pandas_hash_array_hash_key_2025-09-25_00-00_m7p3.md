# Bug Report: pandas.core.util.hashing.hash_array Ignores hash_key for Numeric Arrays

**Target**: `pandas.core.util.hashing.hash_array`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `hash_array` function accepts a `hash_key` parameter but silently ignores it for numeric arrays (int, float, bool, datetime, timedelta), only using it for object arrays. This violates the documented API contract and makes the parameter's behavior inconsistent and confusing.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings


@given(st.text(min_size=16, max_size=16))
@settings(max_examples=500)
def test_hash_array_different_hash_keys(hash_key):
    from pandas.core.util.hashing import hash_array

    arr = np.array([1, 2, 3])
    default_hash = hash_array(arr, hash_key="0123456789123456")
    custom_hash = hash_array(arr, hash_key=hash_key)

    if hash_key == "0123456789123456":
        assert np.array_equal(default_hash, custom_hash)
    else:
        assert not np.array_equal(default_hash, custom_hash)
```

**Failing input**: `hash_key='0000000000000000'` (or any 16-byte string different from default)

## Reproducing the Bug

```python
import numpy as np
from pandas.core.util.hashing import hash_array

arr = np.array([1, 2, 3, 4, 5])

hash1 = hash_array(arr, hash_key="0123456789123456")
hash2 = hash_array(arr, hash_key="AAAAAAAAAAAAAAAA")
hash3 = hash_array(arr, hash_key="different_key123")

print(f"hash_key='0123456789123456': {hash1}")
print(f"hash_key='AAAAAAAAAAAAAAAA': {hash2}")
print(f"hash_key='different_key123': {hash3}")
print(f"\nAll identical: {np.array_equal(hash1, hash2) and np.array_equal(hash2, hash3)}")

obj_arr = np.array(['a', 'b'], dtype=object)
obj_hash1 = hash_array(obj_arr, hash_key="0123456789123456", categorize=False)
obj_hash2 = hash_array(obj_arr, hash_key="AAAAAAAAAAAAAAAA", categorize=False)
print(f"\nObject arrays respect hash_key: {not np.array_equal(obj_hash1, obj_hash2)}")
```

## Why This Is A Bug

1. **API Contract Violation**: The function's docstring documents `hash_key` as a parameter "for string key to encode", but it's silently ignored for numeric dtypes. Users have no way to know when the parameter will or won't be used.

2. **Inconsistent Behavior**: The same parameter works for object arrays but not numeric arrays, creating surprising behavior:
   ```python
   hash_array(np.array([1, 2, 3]), hash_key="custom")      # hash_key ignored
   hash_array(np.array(['a','b','c']), hash_key="custom")  # hash_key used
   ```

3. **No Warning**: The function accepts the parameter without any indication it's being ignored, making it impossible for users to detect they're passing a useless argument.

## Fix

The fix depends on the intended behavior. Two options:

**Option 1: Use hash_key for all dtypes** (recommended for consistency)

```diff
--- a/pandas/core/util/hashing.py
+++ b/pandas/core/util/hashing.py
@@ -298,11 +298,16 @@ def _hash_ndarray(

     # First, turn whatever array this is into unsigned 64-bit ints, if we can
     # manage it.
     if dtype == bool:
         vals = vals.astype("u8")
     elif issubclass(dtype.type, (np.datetime64, np.timedelta64)):
         vals = vals.view("i8").astype("u8", copy=False)
     elif issubclass(dtype.type, np.number) and dtype.itemsize <= 8:
         vals = vals.view(f"u{vals.dtype.itemsize}").astype("u8")
+        # Apply hash_key mixing to numeric values for consistency
+        hash_key_int = int.from_bytes(hash_key.encode(encoding)[:8], 'little')
+        vals ^= np.uint64(hash_key_int)
     else:
         ...
```

**Option 2: Document the limitation** (easier but less user-friendly)

```diff
--- a/pandas/core/util/hashing.py
+++ b/pandas/core/util/hashing.py
@@ -246,7 +246,8 @@ def hash_array(
     encoding : str, default 'utf8'
         Encoding for data & key when strings.
     hash_key : str, default _default_hash_key
-        Hash_key for string key to encode.
+        Hash_key for string key to encode. Only applies to object arrays;
+        ignored for numeric dtypes.
     categorize : bool, default True
         Whether to first categorize object arrays before hashing. This is more
         efficient when the array contains duplicate values.
```