# Bug Report: pandas.core.algorithms.factorize Conflates Empty String and Null Character

**Target**: `pandas.core.algorithms.factorize`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `factorize` function incorrectly treats empty string `''` and null character `'\x00'` as identical values, violating its documented guarantee that `uniques.take(codes)` reconstructs the original values.

## Property-Based Test

```python
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings
import pandas.core.algorithms as algorithms

@given(st.lists(st.one_of(st.integers(), st.floats(allow_nan=False, allow_infinity=False), st.text()), min_size=0, max_size=100))
@settings(max_examples=500)
def test_factorize_round_trip(values):
    """Property: uniques[codes] should reconstruct original values (ignoring NaN sentinel)"""
    try:
        codes, uniques = algorithms.factorize(values)

        reconstructed = []
        for code in codes:
            if code == -1:
                reconstructed.append(np.nan)
            else:
                reconstructed.append(uniques[code])

        for i, (orig, recon) in enumerate(zip(values, reconstructed)):
            if pd.isna(orig) and pd.isna(recon):
                continue
            assert orig == recon, f"Mismatch at index {i}: {orig} != {recon}"
    except (TypeError, ValueError):
        pass
```

<details>

<summary>
**Failing input**: `['', '\x00']`
</summary>
```
/home/npc/pbt/agentic-pbt/worker_/3/hypo.py:33: FutureWarning: factorize with argument that is not not a Series, Index, ExtensionArray, or np.ndarray is deprecated and will raise in a future version.
  codes, uniques = algorithms.factorize(values)
Running Hypothesis test with failing input ['', '\x00']...
AssertionError: Mismatch at index 1: '\x00' != ''

Detailed analysis:
Input: ['', '\x00']
Codes: [0 0]
Uniques: ['']
Reconstructed: ['', '']
```
</details>

## Reproducing the Bug

```python
import pandas.core.algorithms as algorithms
import numpy as np

values = ['', '\x00']
codes, uniques = algorithms.factorize(values)

print(f"Input: {repr(values)}")
print(f"Codes: {codes}")
print(f"Uniques: {uniques}")

for i, code in enumerate(codes):
    recon = uniques[code]
    orig = values[i]
    print(f"Index {i}: orig={repr(orig)}, recon={repr(recon)}, match={orig == recon}")

print("\nAdditional verification:")
print(f"Empty string == null char: {'' == '\x00'}")
print(f"len(''): {len('')}")
print(f"len('\\x00'): {len('\x00')}")
print(f"ord('\\x00'): {ord('\x00')}")

# Verify round-trip property
print("\nRound-trip reconstruction using uniques.take(codes):")
reconstructed = uniques.take(codes)
print(f"Reconstructed: {repr(reconstructed)}")
print(f"Original: {repr(values)}")
print(f"Match: {list(reconstructed) == values}")
```

<details>

<summary>
AssertionError: Values ['', '\x00'] produce identical codes but are distinct
</summary>
```
/home/npc/pbt/agentic-pbt/worker_/3/repo.py:5: FutureWarning: factorize with argument that is not not a Series, Index, ExtensionArray, or np.ndarray is deprecated and will raise in a future version.
  codes, uniques = algorithms.factorize(values)
Input: ['', '\x00']
Codes: [0 0]
Uniques: ['']
Index 0: orig='', recon='', match=True
Index 1: orig='\x00', recon='', match=False

Additional verification:
Empty string == null char: False
len(''): 0
len('\x00'): 1
ord('\x00'): 0

Round-trip reconstruction using uniques.take(codes):
Reconstructed: array(['', ''], dtype=object)
Original: ['', '\x00']
Match: False
```
</details>

## Why This Is A Bug

This violates the explicit documented guarantee in `pandas.core.algorithms.factorize`:

> **Returns**: codes : ndarray
> An integer ndarray that's an indexer into `uniques`. **`uniques.take(codes)` will have the same values as `values`.**

The empty string `''` and null character `'\x00'` are distinct values in Python:
- `'' == '\x00'` evaluates to `False`
- `len('')` is 0, while `len('\x00')` is 1
- The null character has ASCII value 0 (`ord('\x00') == 0`)

Despite being different values, `factorize` assigns both the same code (0) and only includes the empty string in the uniques array. This causes silent data corruption where `'\x00'` is incorrectly reconstructed as `''`, violating the round-trip property that is fundamental to the function's contract.

## Relevant Context

The bug likely originates in the C/Cython implementation of pandas' hash tables, specifically in either `StringHashTable` or `PyObjectHashTable` (defined in `pandas._libs.hashtable`). These implementations may use C-style string handling where null characters are treated as string terminators, causing `'\x00'` to be incorrectly processed as an empty string.

This affects real-world use cases involving:
- Binary data converted to strings
- Network protocols with null-terminated strings
- File format parsers (many formats use null bytes as separators)
- Database exports containing null characters
- Interfacing with C/C++ code that uses null-terminated strings

The issue is found in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/algorithms.py:795` where `factorize_array` is called, which then delegates to the hashtable implementation at line 595.

## Proposed Fix

Since the bug is in the low-level C/Cython hashtable implementation, a complete fix requires modifying the pandas source code. The hash table implementation needs to be updated to properly handle strings containing null characters by using length-aware string comparison rather than C-style null-terminated string functions.

A high-level approach to fix this:

1. In the hashtable implementation (`pandas/_libs/hashtable.pyx`), ensure string comparison uses the full Python string length, not C-style strlen()
2. Update hashing functions to include null characters in the hash computation
3. Ensure string storage preserves the full string including embedded nulls

Without access to modify the pandas source, users can work around this by:
- Pre-processing data to replace null characters with a placeholder before factorization
- Using a different encoding for strings containing null bytes
- Converting to categorical type with explicit categories that distinguish '' and '\x00'