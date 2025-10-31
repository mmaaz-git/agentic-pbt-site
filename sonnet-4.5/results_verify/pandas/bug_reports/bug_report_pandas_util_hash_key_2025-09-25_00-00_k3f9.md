# Bug Report: pandas.util Hash Functions - Undocumented hash_key Length Requirement

**Target**: `pandas.util.hash_pandas_object`, `pandas.util.hash_array`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `hash_pandas_object` and `hash_array` functions accept a `hash_key` parameter documented as a string, but the implementation requires it to be exactly 16 bytes when encoded. This requirement is not documented, leading to confusing `ValueError` exceptions for users.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st
import pandas.util

@given(st.lists(st.integers(), min_size=1, max_size=100))
def test_hash_pandas_object_hash_key_parameter(lst):
    series = pd.Series(lst)
    hash1 = pandas.util.hash_pandas_object(series, hash_key="key1")
    hash2 = pandas.util.hash_pandas_object(series, hash_key="key2")
```

**Failing input**: `lst=[-9_223_372_036_854_775_809]` with `hash_key="key1"`

## Reproducing the Bug

```python
import pandas as pd
import pandas.util

series = pd.Series([1, 2, 3])

pandas.util.hash_pandas_object(series, hash_key="test")
```

This raises:
```
ValueError: key should be a 16-byte string encoded, got b'test' (len 4)
```

The same issue occurs with `hash_array`:

```python
import numpy as np
import pandas.util

arr = np.array([1, 2, 3])
pandas.util.hash_array(arr, hash_key="test")
```

## Why This Is A Bug

The documentation for both functions states:

```
hash_key : str, default _default_hash_key
    Hash_key for string key to encode.
```

This suggests that any string is acceptable, but the underlying Cython implementation (`pandas/_libs/hashing.pyx`) requires the encoded string to be exactly 16 bytes. This is an API contract violation where:

1. The parameter type hint allows any `str`
2. The docstring doesn't mention the length requirement
3. The error message comes from deep in the call stack, making it unclear what went wrong
4. Users have no way to know the requirement without trial and error or reading the source code

## Fix

Option 1: Document the requirement in the docstring:

```diff
diff --git a/pandas/core/util/hashing.py b/pandas/core/util/hashing.py
index 1234567..abcdefg 100644
--- a/pandas/core/util/hashing.py
+++ b/pandas/core/util/hashing.py
@@ -97,7 +97,8 @@ def hash_pandas_object(
     encoding : str, default 'utf8'
         Encoding for data & key when strings.
     hash_key : str, default _default_hash_key
-        Hash_key for string key to encode.
+        Hash_key for string key to encode. Must be exactly 16 bytes when
+        encoded with the specified encoding.
     categorize : bool, default True
         Whether to first categorize object arrays before hashing. This is more
         efficient when the array contains duplicate values.
@@ -245,7 +246,8 @@ def hash_array(
     encoding : str, default 'utf8'
         Encoding for data & key when strings.
     hash_key : str, default _default_hash_key
-        Hash_key for string key to encode.
+        Hash_key for string key to encode. Must be exactly 16 bytes when
+        encoded with the specified encoding.
     categorize : bool, default True
         Whether to first categorize object arrays before hashing. This is more
         efficient when the array contains duplicate values.
```

Option 2 (better): Add validation with a clear error message at the Python level:

```diff
diff --git a/pandas/core/util/hashing.py b/pandas/core/util/hashing.py
index 1234567..abcdefg 100644
--- a/pandas/core/util/hashing.py
+++ b/pandas/core/util/hashing.py
@@ -120,6 +120,11 @@ def hash_pandas_object(
     if hash_key is None:
         hash_key = _default_hash_key
+
+    encoded_key = hash_key.encode(encoding)
+    if len(encoded_key) != 16:
+        raise ValueError(
+            f"hash_key must be exactly 16 bytes when encoded with {encoding}, "
+            f"got {len(encoded_key)} bytes. Use a 16-character ASCII string or "
+            f"adjust for your encoding."
+        )

     if isinstance(obj, ABCMultiIndex):
```