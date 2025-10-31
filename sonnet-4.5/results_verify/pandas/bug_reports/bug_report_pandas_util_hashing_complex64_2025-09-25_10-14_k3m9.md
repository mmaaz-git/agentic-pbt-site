# Bug Report: pandas.core.util.hashing Complex64 Handling

**Target**: `pandas.core.util.hashing._hash_ndarray`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_hash_ndarray` function incorrectly handles `complex64` arrays differently from `complex128` arrays due to an overly-specific type check, leading to inconsistent hashing behavior between complex types.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
import pandas as pd

@given(st.lists(st.complex_numbers(allow_nan=False, allow_infinity=False), min_size=1))
def test_complex_hashing_consistency(values):
    arr64 = np.array(values, dtype=np.complex64)
    arr128 = np.array(values, dtype=np.complex128)

    hash64 = pd.util.hash_array(arr64)
    hash128 = pd.util.hash_array(arr128)

    hash64_real = pd.util.hash_array(arr64.real)
    hash64_imag = pd.util.hash_array(arr64.imag)
    expected64 = hash64_real + 23 * hash64_imag

    assert np.array_equal(hash64, expected64), \
        "complex64 should use same formula as complex128: hash_real + 23 * hash_imag"
```

**Failing input**: `[1+2j]` as `complex64`

## Reproducing the Bug

```python
import numpy as np
import pandas as pd

arr64 = np.array([1+2j, 3+4j], dtype=np.complex64)
arr128 = np.array([1+2j, 3+4j], dtype=np.complex128)

hash64 = pd.util.hash_array(arr64)
hash128 = pd.util.hash_array(arr128)

hash64_real = pd.util.hash_array(arr64.real)
hash64_imag = pd.util.hash_array(arr64.imag)
expected64 = hash64_real + 23 * hash64_imag

print(f"complex64 hash: {hash64}")
print(f"Expected:       {expected64}")
print(f"Match: {np.array_equal(hash64, expected64)}")

print(f"\ncomplex128 hash: {hash128}")
hash128_real = pd.util.hash_array(arr128.real)
hash128_imag = pd.util.hash_array(arr128.imag)
expected128 = hash128_real + 23 * hash128_imag
print(f"Expected:        {expected128}")
print(f"Match: {np.array_equal(hash128, expected128)}")
```

## Why This Is A Bug

In `pandas/core/util/hashing.py` at line 294, the code checks:

```python
if np.issubdtype(dtype, np.complex128):
    hash_real = _hash_ndarray(vals.real, encoding, hash_key, categorize)
    hash_imag = _hash_ndarray(vals.imag, encoding, hash_key, categorize)
    return hash_real + 23 * hash_imag
```

The function `np.issubdtype(dtype, np.complex128)` only returns `True` for `complex128` and its subtypes, NOT for `complex64`. This means:

1. **complex128 arrays**: Caught by special case, hashed as `hash_real + 23 * hash_imag`
2. **complex64 arrays**: Fall through to line 305, treated as numeric with `itemsize <= 8`, viewed as `uint64` directly

This creates inconsistent behavior:
- Same complex values hash differently based on precision (complex64 vs complex128)
- The documented formula (line 297 comment) only applies to complex128
- Violates user expectation that equivalent values should hash the same way

## Fix

```diff
--- a/pandas/core/util/hashing.py
+++ b/pandas/core/util/hashing.py
@@ -291,7 +291,7 @@ def _hash_ndarray(
     dtype = vals.dtype

     # _hash_ndarray only takes 64-bit values, so handle 128-bit by parts
-    if np.issubdtype(dtype, np.complex128):
+    if np.issubdtype(dtype, np.complexfloating):
         hash_real = _hash_ndarray(vals.real, encoding, hash_key, categorize)
         hash_imag = _hash_ndarray(vals.imag, encoding, hash_key, categorize)
         return hash_real + 23 * hash_imag
```

Using `np.complexfloating` catches all complex dtypes (complex64, complex128, and any extended precision complex types), ensuring consistent hashing behavior across all complex number types.