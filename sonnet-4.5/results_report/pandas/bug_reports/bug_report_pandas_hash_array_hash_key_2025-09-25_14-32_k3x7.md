# Bug Report: pandas.core.util.hashing.hash_array Ignores hash_key for Numeric Arrays

**Target**: `pandas.core.util.hashing.hash_array`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `hash_array` function accepts a `hash_key` parameter that is documented to affect the hash output, but this parameter is completely ignored when hashing numeric arrays (int, float, bool, datetime, timedelta). This violates the API contract and makes the parameter misleading.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st
from pandas.core.util.hashing import hash_array


@given(st.lists(st.integers(), min_size=1, max_size=50))
def test_hash_array_key_ignored_for_numeric_arrays(values):
    arr = np.array(values)
    hash_key1 = '0' * 16
    hash_key2 = '1' * 16

    hash1 = hash_array(arr, hash_key=hash_key1, categorize=False)
    hash2 = hash_array(arr, hash_key=hash_key2, categorize=False)

    assert not np.array_equal(hash1, hash2), \
        "Different hash keys should produce different hashes for numeric arrays"
```

**Failing input**: `[0]`

## Reproducing the Bug

```python
import numpy as np
from pandas.core.util.hashing import hash_array

arr = np.array([1, 2, 3])

hash1 = hash_array(arr, hash_key='0' * 16)
hash2 = hash_array(arr, hash_key='1' * 16)

print("Hash with key '0'*16:", hash1)
print("Hash with key '1'*16:", hash2)
print("Are they equal?:", np.array_equal(hash1, hash2))
```

**Output:**
```
Hash with key '0'*16: [ 6238072747940578789 15839785061582574730  2185194620014831856]
Hash with key '1'*16: [ 6238072747940578789 15839785061582574730  2185194620014831856]
Are they equal?: True
```

## Why This Is A Bug

The function signature and docstring both specify that `hash_key` is a parameter that should affect the hash:

```python
def hash_array(
    vals: ArrayLike,
    encoding: str = "utf8",
    hash_key: str = _default_hash_key,  # Parameter exists
    categorize: bool = True,
) -> npt.NDArray[np.uint64]:
    """
    ...
    hash_key : str, default _default_hash_key
        Hash_key for string key to encode.  # Documented
    ...
    """
```

However, examining `_hash_ndarray` (called by `hash_array`), numeric arrays bypass the hash_key entirely:

```python
# From pandas/core/util/hashing.py, line ~300
elif issubclass(dtype.type, np.number) and dtype.itemsize <= 8:
    vals = vals.view(f"u{vals.dtype.itemsize}").astype("u8")
# ... then applies bitwise operations WITHOUT using hash_key
```

The hash_key is only used when:
1. Arrays have object dtype (strings)
2. Arrays are categorized and have object categories

This violates the API contract: users passing different hash_keys expect different hashes, but for numeric arrays they get identical results.

## Fix

The fix requires incorporating the hash_key into the numeric array hashing path. One approach:

```diff
diff --git a/pandas/core/util/hashing.py b/pandas/core/util/hashing.py
index abc123..def456 100644
--- a/pandas/core/util/hashing.py
+++ b/pandas/core/util/hashing.py
@@ -338,6 +338,10 @@ def _hash_ndarray(
     vals *= np.uint64(0x94D049BB133111EB)
     vals ^= vals >> 31
+
+    # Incorporate hash_key for consistency with object arrays
+    key_hash = hash_object_array(np.array([hash_key], dtype=object), hash_key, encoding)[0]
+    vals ^= key_hash
+
     return vals
```

Alternatively, the function could validate that hash_key is only provided when it will be used, or update documentation to clarify the limitation.