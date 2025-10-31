# Bug Report: pandas.core.algorithms.factorize Null Character String Collision

**Target**: `pandas.core.algorithms.factorize`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `factorize` function incorrectly treats strings containing null characters (`\x00`) as identical to empty strings or other null-character strings, violating its documented contract that "uniques.take(codes) will have the same values as values".

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from pandas.core.algorithms import factorize

@given(st.lists(st.text(min_size=0, max_size=10)))
@settings(max_examples=1000)
def test_factorize_roundtrip_strings(values):
    values_array = np.array(values, dtype=object)
    codes, uniques = factorize(values_array)

    reconstructed = uniques.take(codes[codes >= 0])
    original_non_nan = values_array[codes >= 0]

    assert len(reconstructed) == len(original_non_nan)
    assert np.array_equal(reconstructed, original_non_nan)

if __name__ == "__main__":
    test_factorize_roundtrip_strings()
```

<details>

<summary>
**Failing input**: `values=['', '\x00']`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 18, in <module>
    test_factorize_roundtrip_strings()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 6, in test_factorize_roundtrip_strings
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 15, in test_factorize_roundtrip_strings
    assert np.array_equal(reconstructed, original_non_nan)
           ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_factorize_roundtrip_strings(
    values=['', '\x00'],
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from pandas.core.algorithms import factorize

# Test case with empty string and null character
values = np.array(['', '\x00'], dtype=object)
codes, uniques = factorize(values)

print(f"Input: {[repr(v) for v in values]}")
print(f"Codes: {codes}")
print(f"Uniques: {[repr(v) for v in uniques]}")

# Attempt to reconstruct original values
reconstructed = uniques.take(codes)
print(f"Reconstructed: {[repr(v) for v in reconstructed]}")
print(f"Match original? {np.array_equal(reconstructed, values)}")

# Additional test cases
print("\n--- Additional test cases ---")

# Test: single null vs double null
values2 = np.array(['\x00', '\x00\x00'], dtype=object)
codes2, uniques2 = factorize(values2)
print(f"\nInput: {[repr(v) for v in values2]}")
print(f"Codes: {codes2}")
print(f"Uniques: {[repr(v) for v in uniques2]}")
print(f"Reconstructed matches? {np.array_equal(uniques2.take(codes2), values2)}")

# Test: empty, null, and regular char
values3 = np.array(['', '\x00', 'a'], dtype=object)
codes3, uniques3 = factorize(values3)
print(f"\nInput: {[repr(v) for v in values3]}")
print(f"Codes: {codes3}")
print(f"Uniques: {[repr(v) for v in uniques3]}")
print(f"Reconstructed matches? {np.array_equal(uniques3.take(codes3), values3)}")

# Test that works correctly: empty and \x01
values4 = np.array(['', '\x01'], dtype=object)
codes4, uniques4 = factorize(values4)
print(f"\nInput: {[repr(v) for v in values4]}")
print(f"Codes: {codes4}")
print(f"Uniques: {[repr(v) for v in uniques4]}")
print(f"Reconstructed matches? {np.array_equal(uniques4.take(codes4), values4)}")
```

<details>

<summary>
Distinct strings are incorrectly mapped to the same code
</summary>
```
Input: ["''", "'\\x00'"]
Codes: [0 0]
Uniques: ["''"]
Reconstructed: ["''", "''"]
Match original? False

--- Additional test cases ---

Input: ["'\\x00'", "'\\x00\\x00'"]
Codes: [0 0]
Uniques: ["'\\x00'"]
Reconstructed matches? False

Input: ["''", "'\\x00'", "'a'"]
Codes: [0 0 1]
Uniques: ["''", "'a'"]
Reconstructed matches? False

Input: ["''", "'\\x01'"]
Codes: [0 1]
Uniques: ["''", "'\\x01'"]
Reconstructed matches? True
```
</details>

## Why This Is A Bug

This violates the explicitly documented contract of `factorize`. The function's docstring states: "codes : ndarray - An integer ndarray that's an indexer into `uniques`. `uniques.take(codes)` will have the same values as `values`." This guarantee is broken when strings contain null characters.

The bug causes **silent data corruption**: Two distinct Python string values (`''` with length 0 and `'\x00'` with length 1) are assigned the same code, making them indistinguishable after factorization. This violates Python's string semantics where these are different values (`'' != '\x00'`).

The issue appears to stem from underlying C code treating `\x00` as a string terminator rather than a valid character within Python strings, which are length-prefixed and can contain null bytes.

## Relevant Context

- **Similar known issue**: GitHub issue #34551 confirms related null byte handling problems in pandas StringHashTable
- **Affected use cases**: Binary data in strings, C-string imports with embedded nulls, control characters in text data, database values with null bytes
- **Inconsistent behavior**: The function correctly distinguishes `''` from `'\x01'` but fails for `'\x00'`, indicating special (incorrect) handling of null bytes
- **Documentation reference**: [pandas.factorize documentation](https://pandas.pydata.org/docs/reference/api/pandas.factorize.html)
- **Python string semantics**: Python strings can contain any byte value including nulls - [PEP 393](https://peps.python.org/pep-0393/)

## Proposed Fix

The fix requires ensuring that string comparison and hashing in the factorization code properly handles the full string content including null bytes, not treating `\x00` as a terminator. This likely involves modifying the underlying StringHashTable implementation to use Python's string length rather than relying on C-style null termination. A high-level approach would be:

1. Identify where StringHashTable or factorize_array uses C string functions
2. Replace strlen/strcmp with Python string operations that respect the full string length
3. Ensure hash functions consider all bytes in the string, not stopping at null bytes
4. Add comprehensive test coverage for strings with null characters

Without access to the exact implementation details, a specific patch cannot be provided, but the core issue is clear: the code must respect Python's string semantics where null bytes are valid characters, not terminators.