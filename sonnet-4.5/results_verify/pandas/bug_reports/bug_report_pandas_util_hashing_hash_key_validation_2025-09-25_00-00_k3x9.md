# Bug Report: pandas.core.util.hashing hash_key Validation

**Target**: `pandas.core.util.hashing.hash_array` (also affects `hash_pandas_object` and `hash_tuples`)
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `hash_key` parameter in hashing functions requires exactly 16 bytes when encoded, but this constraint is not documented and validation only occurs deep in the implementation, producing a confusing error message.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
import pandas.core.util.hashing as hashing

@given(
    st.lists(st.integers(), min_size=1, max_size=100),
    st.text(min_size=1, max_size=20),
)
def test_hash_array_different_hash_keys(values, hash_key):
    arr = np.array(values)
    hash1 = hashing.hash_array(arr, hash_key="key1")
    hash2 = hashing.hash_array(arr, hash_key="key2")
    if hash_key != "key2":
        assert not np.array_equal(hash1, hash2)
```

**Failing input**: `values=[-9_223_372_036_854_775_809], hash_key='0'`

## Reproducing the Bug

```python
import numpy as np
import pandas.core.util.hashing as hashing

arr = np.array(["hello"], dtype=object)
result = hashing.hash_array(arr, hash_key="short")
```

Output:
```
ValueError: key should be a 16-byte string encoded, got b'short' (len 5)
```

## Why This Is A Bug

1. The documentation for `hash_key` parameter states: "Hash_key for string key to encode" without mentioning the 16-byte requirement
2. The function signature shows a default value but doesn't hint at the constraint: `hash_key: str = '0123456789123456'`
3. The error occurs deep in the Cython implementation (`pandas/_libs/hashing.pyx`) rather than at the public API boundary
4. Users have no way to know the constraint exists unless they encounter this error

This violates the API contract - the public interface should either:
- Document the constraint clearly, or
- Validate the constraint early with a clear error message, or
- Accept any-length strings and adapt accordingly

## Fix

Add validation at the public API level with a clear error message:

```diff
diff --git a/pandas/core/util/hashing.py b/pandas/core/util/hashing.py
index 1234567890..abcdefghij 100644
--- a/pandas/core/util/hashing.py
+++ b/pandas/core/util/hashing.py
@@ -240,6 +240,11 @@ def hash_array(
     >>> pd.util.hash_array(np.array([1, 2, 3]))
     array([ 6238072747940578789, 15839785061582574730,  2185194620014831856],
       dtype=uint64)
     """
+    # Validate hash_key length for object dtype arrays
+    encoded_key = hash_key.encode(encoding)
+    if len(encoded_key) != 16:
+        raise ValueError(
+            f"hash_key must be exactly 16 bytes when encoded with {encoding}, "
+            f"got {len(encoded_key)} bytes. Please provide a string that encodes to 16 bytes."
+        )
+
     if not hasattr(vals, "dtype"):
         raise TypeError("must pass a ndarray-like")
```

Alternatively, update the docstring to document this requirement:

```diff
diff --git a/pandas/core/util/hashing.py b/pandas/core/util/hashing.py
index 1234567890..abcdefghij 100644
--- a/pandas/core/util/hashing.py
+++ b/pandas/core/util/hashing.py
@@ -226,7 +226,8 @@ def hash_array(
 encoding : str, default 'utf8'
     Encoding for data & key when strings.
 hash_key : str, default _default_hash_key
-    Hash_key for string key to encode.
+    Hash_key for string key to encode. Must be exactly 16 bytes when
+    encoded with the specified encoding.
 categorize : bool, default True
     Whether to first categorize object arrays before hashing. This is more
     efficient when the array contains duplicate values.
```