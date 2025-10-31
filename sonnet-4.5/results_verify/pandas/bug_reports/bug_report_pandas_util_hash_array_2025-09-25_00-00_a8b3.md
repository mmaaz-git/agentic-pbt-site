# Bug Report: pandas.core.util.hashing.hash_array Multi-byte Hash Key Encoding

**Target**: `pandas.core.util.hashing.hash_array`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`hash_array()` crashes with a ValueError when given a 16-character hash_key containing non-ASCII characters that encode to more than 16 bytes in UTF-8.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from pandas.core.util.hashing import hash_array

@given(st.lists(st.text(max_size=10), min_size=1),
       st.text(min_size=16, max_size=16))
def test_hash_array_object_different_hash_keys(str_list, hash_key1):
    arr = np.array(str_list, dtype=object)
    hash_key2 = hash_key1[:15] + ('x' if hash_key1[15] != 'x' else 'y')

    result1 = hash_array(arr, hash_key=hash_key1)
    result2 = hash_array(arr, hash_key=hash_key2)

    assert not np.array_equal(result1, result2)
```

**Failing input**: `str_list=['']`, `hash_key1='000000000000000\x80'`

## Reproducing the Bug

```python
import numpy as np
from pandas.core.util.hashing import hash_array

arr = np.array([''], dtype=object)
hash_key = '000000000000000\x80'

print(f"hash_key length (characters): {len(hash_key)}")
print(f"hash_key encoded length (bytes): {len(hash_key.encode('utf8'))}")

result = hash_array(arr, hash_key=hash_key)
```

**Output:**
```
hash_key length (characters): 16
hash_key encoded length (bytes): 17
ValueError: key should be a 16-byte string encoded, got b'000000000000000\xc2\x80' (len 17)
```

## Why This Is A Bug

The function accepts a `hash_key` parameter as a string but doesn't validate that it encodes to exactly 16 bytes. The underlying C implementation (`hash_object_array`) requires a 16-byte encoded key, but the Python API accepts any 16-character string. When a user provides a 16-character string with non-ASCII characters (which is valid Python), it encodes to more than 16 bytes in UTF-8, causing a crash.

The docstring doesn't specify this requirement:
```python
hash_key : str, default _default_hash_key
    Hash_key for string key to encode.
```

A reasonable user would expect any 16-character string to work since the default `_default_hash_key = "0123456789123456"` is 16 characters.

## Fix

Add validation to ensure the hash_key encodes to exactly 16 bytes:

```diff
diff --git a/pandas/core/util/hashing.py b/pandas/core/util/hashing.py
index 1234567..abcdefg 100644
--- a/pandas/core/util/hashing.py
+++ b/pandas/core/util/hashing.py
@@ -236,6 +236,10 @@ def hash_array(
     hash_key : str, default _default_hash_key
         Hash_key for string key to encode.
+        Must be a string that encodes to exactly 16 bytes using the specified encoding.
     categorize : bool, default True
@@ -262,6 +266,11 @@ def hash_array(
     """
     if not hasattr(vals, "dtype"):
         raise TypeError("must pass a ndarray-like")
+
+    if len(hash_key.encode(encoding)) != 16:
+        raise ValueError(
+            f"hash_key must encode to exactly 16 bytes, got {len(hash_key.encode(encoding))} bytes"
+        )

     if isinstance(vals, ABCExtensionArray):
         return vals._hash_pandas_object(
```

Alternatively, update the docstring to clearly state the requirement and provide an example of what happens with multi-byte characters.