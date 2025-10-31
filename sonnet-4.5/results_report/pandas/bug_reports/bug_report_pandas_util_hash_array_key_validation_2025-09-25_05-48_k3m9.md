# Bug Report: pandas.util.hash_array Inadequate hash_key Validation

**Target**: `pandas.util.hash_array`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `hash_array` function accepts a `hash_key` parameter but does not validate its length upfront. The function fails deep in C code with a cryptic error message when the hash_key is not exactly 16 bytes (when UTF-8 encoded), violating the principle of early validation and clear error reporting.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
import pandas.util


@given(st.text(min_size=1, max_size=15))
def test_hash_array_validates_hash_key_length(hash_key):
    assume(len(hash_key.encode('utf8')) < 16)
    arr = np.array(['a', 'b', 'c'], dtype=object)

    result = pandas.util.hash_array(arr, hash_key=hash_key)
```

**Failing input**: Any string whose UTF-8 encoding is not exactly 16 bytes, e.g., `hash_key='short'`

## Reproducing the Bug

```python
import numpy as np
import pandas.util

arr = np.array(['a', 'b', 'c'], dtype=object)

pandas.util.hash_array(arr, hash_key="short")
```

Output:
```
ValueError: key should be a 16-byte string encoded, got b'short' (len 5)
```

The error occurs deep in the Cython implementation (`pandas/_libs/hashing.pyx:63`), not at the Python API level.

## Why This Is A Bug

This violates several software engineering best practices:

1. **Late validation**: The function accepts the parameter without validation and only fails when it reaches low-level C code
2. **Poor error location**: The error is raised from a deep call stack in Cython, making it hard to debug
3. **Cryptic error message**: Users see internal implementation details (byte encoding) rather than a clear API-level message
4. **Inconsistent with pandas conventions**: Other pandas functions typically validate parameters early with clear messages

The `hash_key` parameter is accepted for all array dtypes, but:
- For numeric arrays, it's silently ignored (doesn't cause an error)
- For object/string arrays, it fails with a cryptic low-level error if not exactly 16 bytes

## Fix

Add early validation in the `hash_array` function:

```diff
def hash_array(
    vals: ArrayLike,
    encoding: str = "utf8",
    hash_key: str = _default_hash_key,
    categorize: bool = True,
) -> npt.NDArray[np.uint64]:
    """
    Given a 1d array, return an array of deterministic integers.
    ...
    """
    if not hasattr(vals, "dtype"):
        raise TypeError("must pass a ndarray-like")

+   # Validate hash_key early with clear error message
+   if hash_key is not None:
+       hash_key_bytes = hash_key.encode(encoding)
+       if len(hash_key_bytes) != 16:
+           raise ValueError(
+               f"hash_key must be exactly 16 bytes when encoded with '{encoding}', "
+               f"got {len(hash_key_bytes)} bytes from hash_key={hash_key!r}"
+           )

    if isinstance(vals, ABCExtensionArray):
        return vals._hash_pandas_object(
            encoding=encoding, hash_key=hash_key, categorize=categorize
        )
```

Alternatively, update the docstring to clearly document that:
1. hash_key must be exactly 16 bytes when UTF-8 encoded
2. hash_key only applies to object/string dtypes (ignored for numeric)