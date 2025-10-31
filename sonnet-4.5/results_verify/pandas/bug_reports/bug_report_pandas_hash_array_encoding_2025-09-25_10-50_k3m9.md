# Bug Report: pandas.core.util.hashing.hash_array Encoding Incompatibility

**Target**: `pandas.core.util.hashing.hash_array`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `hash_array` function fails when using non-UTF-8 encodings (like UTF-16) with the default `hash_key` parameter, even though the encoding parameter is documented as valid. The function requires `hash_key` to be exactly 16 bytes when encoded, but this requirement is not documented and incompatible with UTF-16 encoding.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from pandas.core.util.hashing import hash_array


@settings(max_examples=100)
@given(st.lists(st.integers(), min_size=1, max_size=100))
def test_hash_array_different_encodings(values):
    arr = np.array([str(v) for v in values], dtype=object)
    result_utf8 = hash_array(arr, encoding='utf8')
    result_utf16 = hash_array(arr, encoding='utf16')
    assert len(result_utf8) == len(arr)
    assert len(result_utf16) == len(arr)
```

**Failing input**: `values=[0]` (or any list of integers)

## Reproducing the Bug

```python
import numpy as np
from pandas.core.util.hashing import hash_array

arr_obj = np.array(['a', 'b', 'c'], dtype=object)

result_utf8 = hash_array(arr_obj, encoding='utf8', hash_key='0123456789123456')
print(f"UTF-8: {result_utf8}")

result_utf16 = hash_array(arr_obj, encoding='utf16', hash_key='0123456789123456')
```

**Output**:
```
ValueError: key should be a 16-byte string encoded, got b'\xff\xfe0\x001\x002\x003\x004\x005\x006\x007\x008\x009\x001\x002\x003\x004\x005\x006\x00' (len 34)
```

## Why This Is A Bug

1. **Inconsistent behavior**: The default `hash_key='0123456789123456'` works with UTF-8 but fails with UTF-16
2. **Undocumented requirement**: The docstring doesn't mention that `hash_key` must be exactly 16 bytes when encoded
3. **API contract violation**: The function signature accepts `encoding` as a parameter, implying it should work with any valid encoding, but UTF-16 encoding makes the default hash_key invalid
4. **Counter-intuitive**: The validity of `hash_key` depends on the `encoding` parameter, which is unexpected

The root cause is that:
- UTF-8 encoding: `'0123456789123456'.encode('utf8')` = 16 bytes ✓
- UTF-16 encoding: `'0123456789123456'.encode('utf16')` = 34 bytes (2 bytes BOM + 2 bytes per char) ✗

## Fix

Add validation and better documentation. Either:

**Option 1**: Validate and provide a clear error message upfront
```diff
diff --git a/pandas/core/util/hashing.py b/pandas/core/util/hashing.py
index abc123..def456 100644
--- a/pandas/core/util/hashing.py
+++ b/pandas/core/util/hashing.py
@@ -245,6 +245,11 @@ def hash_array(
     encoding : str, default 'utf8'
         Encoding for data & key when strings.
+        Note: hash_key must be exactly 16 bytes when encoded with this encoding.
     hash_key : str, default _default_hash_key
         Hash_key for string key to encode.
+        Must be exactly 16 bytes when encoded with the specified encoding.
+        The default key '0123456789123456' is 16 bytes in UTF-8 but not in UTF-16.
```

**Option 2**: Make hash_key encoding-independent by always encoding it with UTF-8 regardless of the data encoding parameter

The function should either validate the hash_key length upfront or ensure the hash_key encoding is independent of the data encoding parameter.