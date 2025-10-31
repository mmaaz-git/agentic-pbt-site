# Bug Report: pandas.util.hash_array Inconsistent hash_key Validation

**Target**: `pandas.util.hash_array`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `hash_array` function accepts a `hash_key` parameter but validates it inconsistently. For numeric arrays, any `hash_key` value is silently accepted (and ignored), while for object arrays, it must be exactly 16 bytes or a ValueError is raised.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings, assume
import pandas.util


@given(
    st.lists(st.integers(min_value=-2**63, max_value=2**63-1), min_size=1),
    st.text(min_size=16, max_size=16, alphabet=st.characters()),
    st.text(min_size=16, max_size=16, alphabet=st.characters())
)
@settings(max_examples=500)
def test_hash_array_different_keys_produce_different_hashes(values, key1, key2):
    assume(key1 != key2)
    arr = np.array(values, dtype=np.int64)
    hash1 = pandas.util.hash_array(arr, hash_key=key1)
    hash2 = pandas.util.hash_array(arr, hash_key=key2)

    if len(arr) > 1:
        assert not np.array_equal(hash1, hash2)
```

**Failing input**: When testing with object arrays instead of numeric arrays, the test fails with ValueError for non-16-byte keys.

## Reproducing the Bug

```python
import numpy as np
import pandas as pd

arr_numeric = np.array([1, 2, 3], dtype=np.int64)
arr_object = np.array(['a', 'b', 'c'], dtype=object)

invalid_key = "short_key"

pd.util.hash_array(arr_numeric, hash_key=invalid_key)

try:
    pd.util.hash_array(arr_object, hash_key=invalid_key)
except ValueError as e:
    print(f"ValueError: {e}")
```

## Why This Is A Bug

The function has inconsistent validation behavior that depends on the input array's dtype:

1. **Numeric arrays** (int, float, bool): Accept any `hash_key` value, even though the parameter is ignored for these types
2. **Object arrays** (strings, etc.): Require `hash_key` to be exactly 16 bytes when encoded

This violates API design principles:
- Parameter validation should be consistent and dtype-independent
- Invalid parameters should be rejected upfront, not conditionally based on input type
- The docstring doesn't document the 16-byte requirement
- Users get confusing errors only when using certain data types

## Fix

```diff
--- a/pandas/core/util/hashing.py
+++ b/pandas/core/util/hashing.py
@@ -249,6 +249,11 @@ def hash_array(
     """
     if not hasattr(vals, "dtype"):
         raise TypeError("must pass a ndarray-like")
+
+    if hash_key != _default_hash_key:
+        encoded_key = hash_key.encode('utf8')
+        if len(encoded_key) != 16:
+            raise ValueError(f"hash_key must be exactly 16 bytes when encoded, got {len(encoded_key)} bytes")

     if isinstance(vals, ABCExtensionArray):
         return vals._hash_pandas_object(