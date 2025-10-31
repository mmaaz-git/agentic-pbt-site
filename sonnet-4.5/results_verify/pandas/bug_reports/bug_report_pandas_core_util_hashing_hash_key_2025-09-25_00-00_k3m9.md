# Bug Report: pandas.core.util.hashing Undocumented hash_key Length Requirement

**Target**: `pandas.core.util.hashing.hash_array` and `pandas.core.util.hashing.hash_pandas_object`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `hash_key` parameter in `hash_array()` and `hash_pandas_object()` must be exactly 16 bytes when UTF-8 encoded, but this requirement is undocumented and not validated at the API level. This causes confusing ValueError crashes deep in the call stack when users provide hash keys of different lengths.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from pandas.core.util.hashing import hash_array

@given(st.text(min_size=1))
def test_hash_key_parameter_changes_hash(text):
    arr = np.array([text], dtype=object)
    hash1 = hash_array(arr, hash_key="key1")
    hash2 = hash_array(arr, hash_key="key2")

    assert len(hash1) == 1
    assert len(hash2) == 1
```

**Failing input**: `text='0', hash_key="key1"` (any non-16-byte hash_key will fail)

## Reproducing the Bug

```python
import numpy as np
from pandas.core.util.hashing import hash_array

arr = np.array(["test"], dtype=object)
hash_array(arr, hash_key="short")
```

Output:
```
ValueError: key should be a 16-byte string encoded, got b'short' (len 5)
```

Additional edge case with multi-byte UTF-8:
```python
import numpy as np
from pandas.core.util.hashing import hash_array

arr = np.array(["test"], dtype=object)
hash_array(arr, hash_key="000000000000000🦄")
```

Output:
```
ValueError: key should be a 16-byte string encoded, got b'000000000000000\xf0\x9f\xa6\x84' (len 19)
```

## Why This Is A Bug

The function's docstring specifies:
```
hash_key : str, default _default_hash_key
    Hash_key for string key to encode.
```

This documentation:
1. Does not mention the 16-byte length requirement
2. Does not specify that it must be ASCII/single-byte characters
3. Suggests any string is acceptable

The default value `_default_hash_key = "0123456789123456"` is 16 bytes, but this pattern is not documented as a requirement. Users have no way to know their custom hash_key must follow this constraint until they encounter a cryptic error deep in the call stack.

This violates the principle of least surprise and the API contract implied by the documentation.

## Fix

Add validation at the API level and update documentation:

```diff
diff --git a/pandas/core/util/hashing.py b/pandas/core/util/hashing.py
index 1234567..abcdefg 100644
--- a/pandas/core/util/hashing.py
+++ b/pandas/core/util/hashing.py
@@ -236,6 +236,9 @@ def hash_array(
     vals : ndarray or ExtensionArray
     encoding : str, default 'utf8'
         Encoding for data & key when strings.
     hash_key : str, default _default_hash_key
-        Hash_key for string key to encode.
+        Hash_key for string key to encode. Must be exactly 16 bytes when
+        encoded with the specified encoding.
     categorize : bool, default True
@@ -262,6 +265,12 @@ def hash_array(
     """
     if not hasattr(vals, "dtype"):
         raise TypeError("must pass a ndarray-like")
+
+    encoded_key = hash_key.encode(encoding)
+    if len(encoded_key) != 16:
+        raise ValueError(
+            f"hash_key must be exactly 16 bytes when encoded, "
+            f"got {len(encoded_key)} bytes"
+        )

     if isinstance(vals, ABCExtensionArray):
```

Similar changes should be applied to `hash_pandas_object()` and its docstring.