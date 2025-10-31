# Bug Report: pandas.core.util.hashing Complex64 Type Check Causing Inconsistent Hash Behavior

**Target**: `pandas.core.util.hashing._hash_ndarray`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_hash_ndarray` function incorrectly handles complex64 arrays differently from complex128 arrays due to an overly-specific dtype check, causing mathematically identical complex values to produce different hashes based solely on their precision level.

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

if __name__ == "__main__":
    test_complex_hashing_consistency()
```

<details>

<summary>
**Failing input**: `[complex(0.0, 1.0)]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 21, in <module>
    test_complex_hashing_consistency()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 6, in test_complex_hashing_consistency
    def test_complex_hashing_consistency(values):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 17, in test_complex_hashing_consistency
    assert np.array_equal(hash64, expected64), \
           ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^
AssertionError: complex64 should use same formula as complex128: hash_real + 23 * hash_imag
Falsifying example: test_complex_hashing_consistency(
    values=[complex(0.0, 1.0)],
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import pandas as pd

# Create complex arrays with the same values but different dtypes
arr64 = np.array([1+2j, 3+4j], dtype=np.complex64)
arr128 = np.array([1+2j, 3+4j], dtype=np.complex128)

# Hash both arrays
hash64 = pd.util.hash_array(arr64)
hash128 = pd.util.hash_array(arr128)

# Calculate what the hash64 SHOULD be if it used the same formula as hash128
hash64_real = pd.util.hash_array(arr64.real)
hash64_imag = pd.util.hash_array(arr64.imag)
expected64 = hash64_real + 23 * hash64_imag

print("Complex64 array:", arr64)
print(f"complex64 hash:  {hash64}")
print(f"Expected hash:   {expected64}")
print(f"Match: {np.array_equal(hash64, expected64)}")

print("\nComplex128 array:", arr128)
print(f"complex128 hash: {hash128}")

# Verify complex128 uses the expected formula
hash128_real = pd.util.hash_array(arr128.real)
hash128_imag = pd.util.hash_array(arr128.imag)
expected128 = hash128_real + 23 * hash128_imag
print(f"Expected hash:   {expected128}")
print(f"Match: {np.array_equal(hash128, expected128)}")

print("\nThe bug: complex64 and complex128 arrays with identical values hash differently!")
print(f"Same values? {np.array_equal(arr64, arr128)}")
print(f"Same hashes? {np.array_equal(hash64, hash128)}")
```

<details>

<summary>
Output showing inconsistent hash behavior between complex64 and complex128
</summary>
```
Complex64 array: [1.+2.j 3.+4.j]
complex64 hash:  [14559484286230537508 13875018978056565111]
Expected hash:   [1566622401970978605 9473140185629080060]
Match: False

Complex128 array: [1.+2.j 3.+4.j]
complex128 hash: [15878784018407640025 17123455195525397729]
Expected hash:   [15878784018407640025 17123455195525397729]
Match: True

The bug: complex64 and complex128 arrays with identical values hash differently!
Same values? True
Same hashes? False
```
</details>

## Why This Is A Bug

The bug occurs in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/util/hashing.py` at line 294. The condition `np.issubdtype(dtype, np.complex128)` only returns `True` for complex128 dtypes, excluding complex64.

This violates expected behavior in multiple ways:

1. **Mathematical Consistency**: Mathematically identical complex numbers (e.g., 1+2j) produce completely different hash values based solely on their storage precision. This breaks the fundamental principle that equivalent values should hash the same way.

2. **Documented Intent**: The comment on line 293 states "_hash_ndarray only takes 64-bit values, so handle 128-bit by parts". This suggests the special handling is for values that don't fit in 64 bits. However, the actual purpose is to properly hash ALL complex numbers by decomposing them into real and imaginary components.

3. **Asymmetric Behavior**:
   - complex128 arrays use the formula: `hash(real) + 23 * hash(imag)`
   - complex64 arrays are incorrectly treated as raw 8-byte unsigned integers at line 305-306

4. **Data Integrity Issues**: This inconsistency affects core pandas operations that rely on hashing, including:
   - DataFrame deduplication (drop_duplicates)
   - Groupby operations
   - Merge/join operations
   - Index creation and lookups

## Relevant Context

The type hierarchy shows why the current check fails:
- `np.issubdtype(np.complex64, np.complex128)` returns `False`
- `np.issubdtype(np.complex64, np.complexfloating)` returns `True`
- `np.issubdtype(np.complex128, np.complexfloating)` returns `True`

The code at line 305-306 treats complex64 as a generic numeric type and views it directly as uint64, which completely ignores the semantic structure of complex numbers having distinct real and imaginary parts.

Documentation reference: https://pandas.pydata.org/docs/reference/api/pandas.util.hash_array.html

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