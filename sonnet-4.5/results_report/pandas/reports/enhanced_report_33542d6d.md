# Bug Report: pandas.core.util.hashing Complex64 Inconsistent Hashing

**Target**: `pandas.core.util.hashing.hash_array`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `hash_array` function applies a special hashing formula (`hash(real) + 23 * hash(imag)`) to complex128 arrays but not to complex64 arrays, causing identical complex values with different precisions to produce different hashes.

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

# Run the test
test_hash_array_complex64_vs_complex128()
```

<details>

<summary>
**Failing input**: `values=[1j]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/31/hypo.py", line 20, in <module>
    test_hash_array_complex64_vs_complex128()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/31/hypo.py", line 6, in test_hash_array_complex64_vs_complex128
    @settings(max_examples=500)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/31/hypo.py", line 17, in test_hash_array_complex64_vs_complex128
    assert np.array_equal(hash64, expected64)
           ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_hash_array_complex64_vs_complex128(
    values=[1j],
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from pandas.core.util.hashing import hash_array

# Create complex arrays with the same logical value 1j
c64 = np.array([1j], dtype=np.complex64)
c128 = np.array([1j], dtype=np.complex128)

# Hash both arrays
hash_c64 = hash_array(c64)
hash_c128 = hash_array(c128)

print(f"Complex64 hash: {hash_c64[0]}")
print(f"Complex128 hash: {hash_c128[0]}")
print(f"Hashes are equal: {hash_c64[0] == hash_c128[0]}")

# Calculate expected hash for complex64 using the formula
hash_real_c64 = hash_array(c64.real)
hash_imag_c64 = hash_array(c64.imag)
expected_c64 = hash_real_c64 + 23 * hash_imag_c64

print(f"\nExpected hash for c64 (using formula): {expected_c64[0]}")
print(f"Actual hash for c64: {hash_c64[0]}")
print(f"Match: {hash_c64[0] == expected_c64[0]}")

# Calculate hash for complex128 using the formula
hash_real_c128 = hash_array(c128.real)
hash_imag_c128 = hash_array(c128.imag)
expected_c128 = hash_real_c128 + 23 * hash_imag_c128

print(f"\nExpected hash for c128 (using formula): {expected_c128[0]}")
print(f"Actual hash for c128: {hash_c128[0]}")
print(f"Match: {hash_c128[0] == expected_c128[0]}")
```

<details>

<summary>
Complex64 and complex128 arrays with same value produce different hashes
</summary>
```
Complex64 hash: 2743349149749119347
Complex128 hash: 14479766090982008170
Hashes are equal: False

Expected hash for c64 (using formula): 14869104012228096028
Actual hash for c64: 2743349149749119347
Match: False

Expected hash for c128 (using formula): 14479766090982008170
Actual hash for c128: 14479766090982008170
Match: True
```
</details>

## Why This Is A Bug

This violates the principle of consistent hashing across numeric precisions. The `_hash_ndarray` function at line 294 of `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/util/hashing.py` specifically checks for `np.complex128` and applies the formula `hash(real) + 23 * hash(imag)`. However, complex64 arrays fall through to line 305-306 where they are treated as generic 8-byte numeric values and viewed as uint64, then hashed as a single value.

This creates an inconsistency where:
- Complex128 arrays are decomposed into real and imaginary parts with the formula `hash(real) + 23 * hash(imag)`
- Complex64 arrays are hashed as raw 64-bit unsigned integers without decomposition

This is problematic because:
1. The same logical complex number produces different hashes depending on precision
2. Applications mixing complex64 and complex128 arrays cannot rely on consistent hashing
3. The behavior contradicts the expected consistency for the same logical data type

## Relevant Context

The code comment at line 293 states: "_hash_ndarray only takes 64-bit values, so handle 128-bit by parts". This suggests the special handling was added for size reasons rather than type-specific logic. However, complex64 arrays (8 bytes total: 4 bytes real + 4 bytes imaginary) should logically use the same decomposition approach for consistency.

The current implementation at lines 294-297:
```python
if np.issubdtype(dtype, np.complex128):
    hash_real = _hash_ndarray(vals.real, encoding, hash_key, categorize)
    hash_imag = _hash_ndarray(vals.imag, encoding, hash_key, categorize)
    return hash_real + 23 * hash_imag
```

Complex64 arrays reach lines 305-306 instead:
```python
elif issubclass(dtype.type, np.number) and dtype.itemsize <= 8:
    vals = vals.view(f"u{vals.dtype.itemsize}").astype("u8")
```

This treats the complex64 value as a single 64-bit integer rather than decomposing it.

## Proposed Fix

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