# Bug Report: pandas.core.util.hashing Complex64 Inconsistent Hashing

**Target**: `pandas.core.util.hashing.hash_array`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `hash_array` function handles `complex128` arrays specially by splitting them into real and imaginary parts and using the formula `hash_real + 23 * hash_imag`, but it does not apply the same logic to `complex64` arrays. This creates an inconsistency where equivalent complex numbers in different precisions produce different hashes.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from pandas.core.util.hashing import hash_array

@given(st.lists(st.complex_numbers(allow_nan=False, allow_infinity=False, min_magnitude=1e-10, max_magnitude=1e10), min_size=1))
@settings(max_examples=500)
def test_hash_array_complex64_vs_complex128(values):
    arr64 = np.array(values, dtype=np.complex64)
    arr128 = arr64.astype(np.complex128)

    hash64 = hash_array(arr64)

    real64 = hash_array(arr64.real)
    imag64 = hash_array(arr64.imag)
    expected64 = real64 + 23 * imag64

    assert np.array_equal(hash64, expected64)
```

**Failing input**: `values=[1j]`

## Reproducing the Bug

```python
import numpy as np
from pandas.core.util.hashing import hash_array

c64 = np.array([1j], dtype=np.complex64)
c128 = np.array([1j], dtype=np.complex128)

hash_c64 = hash_array(c64)
hash_c128 = hash_array(c128)

print(f"Complex64 hash: {hash_c64[0]}")
print(f"Complex128 hash: {hash_c128[0]}")

hash_real_c64 = hash_array(c64.real)
hash_imag_c64 = hash_array(c64.imag)
expected_c64 = hash_real_c64 + 23 * hash_imag_c64

print(f"\nExpected hash for c64 (using formula): {expected_c64[0]}")
print(f"Actual hash for c64: {hash_c64[0]}")
print(f"Match: {hash_c64[0] == expected_c64[0]}")
```

Output:
```
Complex64 hash: 2743349149749119347
Complex128 hash: 14479766090982008170

Expected hash for c64 (using formula): 14869104012228096028
Actual hash for c64: 2743349149749119347
Match: False
```

## Why This Is A Bug

The code in `_hash_ndarray` explicitly handles `complex128` by checking `np.issubdtype(dtype, np.complex128)` and then splitting into real/imaginary parts. However, `complex64` arrays (which are 8 bytes) fall through to the generic numeric path where they are viewed as `uint64` and hashed as a single value.

This creates an inconsistency:
- `complex128` arrays are hashed using the formula `hash(real) + 23 * hash(imag)`
- `complex64` arrays are hashed as if they were a single 64-bit unsigned integer

This violates the principle that the hash algorithm should be consistent across different precision types for the same logical data type (complex numbers).

## Fix

```diff
--- a/pandas/core/util/hashing.py
+++ b/pandas/core/util/hashing.py
@@ -240,7 +240,7 @@ def _hash_ndarray(
     dtype = vals.dtype

     # _hash_ndarray only takes 64-bit values, so handle 128-bit by parts
-    if np.issubdtype(dtype, np.complex128):
+    if np.issubdtype(dtype, np.complexfloating):
         hash_real = _hash_ndarray(vals.real, encoding, hash_key, categorize)
         hash_imag = _hash_ndarray(vals.imag, encoding, hash_key, categorize)
         return hash_real + 23 * hash_imag
```

The fix changes the check from `np.complex128` to `np.complexfloating`, which matches all complex floating-point dtypes (including both `complex64` and `complex128`). This ensures that all complex arrays are hashed consistently using the real/imaginary split formula.