# Bug Report: pandas.core.util.hashing.hash_array Hash Collision with Null Bytes

**Target**: `pandas.core.util.hashing.hash_array`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `hash_array` function incorrectly produces hash collisions when `categorize=True`, causing distinct string values containing null bytes to hash to the same value, violating the fundamental property of hash functions.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from pandas.core.util.hashing import hash_array


@settings(max_examples=500)
@given(st.lists(st.text(min_size=0, max_size=10), min_size=1, max_size=20))
def test_hash_array_categorize_equivalence_strings(data):
    arr = np.array(data, dtype=object)
    hash_with_categorize = hash_array(arr, categorize=True)
    hash_without_categorize = hash_array(arr, categorize=False)

    assert np.array_equal(hash_with_categorize, hash_without_categorize)


if __name__ == "__main__":
    test_hash_array_categorize_equivalence_strings()
```

<details>

<summary>
**Failing input**: `data=['', '\x00']`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/46/hypo.py", line 17, in <module>
    test_hash_array_categorize_equivalence_strings()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/46/hypo.py", line 7, in test_hash_array_categorize_equivalence_strings
    @given(st.lists(st.text(min_size=0, max_size=10), min_size=1, max_size=20))
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/46/hypo.py", line 13, in test_hash_array_categorize_equivalence_strings
    assert np.array_equal(hash_with_categorize, hash_without_categorize)
           ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_hash_array_categorize_equivalence_strings(
    data=['', '\x00'],
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from pandas.core.util.hashing import hash_array
from pandas import factorize

# Test case 1: Basic hash collision demonstration
print("=" * 60)
print("Test 1: Basic Hash Collision with Empty String and Null Byte")
print("=" * 60)

data = ['', '\x00']
arr = np.array(data, dtype=object)

hash_with_categorize = hash_array(arr, categorize=True)
hash_without_categorize = hash_array(arr, categorize=False)

print(f"Input data: {repr(data)}")
print(f"Input array dtype: {arr.dtype}")
print()
print(f"Hash with categorize=True:  {hash_with_categorize}")
print(f"Hash with categorize=False: {hash_without_categorize}")
print()

# Check for hash collision
if hash_with_categorize[0] == hash_with_categorize[1]:
    print("❌ BUG CONFIRMED: Empty string '' and null byte '\\x00' have the SAME hash when categorize=True")
else:
    print("✓ No collision detected with categorize=True")

if hash_without_categorize[0] != hash_without_categorize[1]:
    print("✓ Correct: Empty string '' and null byte '\\x00' have DIFFERENT hashes when categorize=False")
else:
    print("❌ Unexpected: Hash collision even with categorize=False")

# Test case 2: Root cause analysis - factorize behavior
print("\n" + "=" * 60)
print("Test 2: Root Cause - factorize() Behavior")
print("=" * 60)

test_data = ['', '\x00', '\x00\x00', 'a', '\x00b']
codes, categories = factorize(test_data)

print(f"Input strings: {repr(test_data)}")
print(f"Factorize codes: {codes}")
print(f"Unique categories: {list(categories)}")
print()
print("Mapping:")
for i, val in enumerate(test_data):
    print(f"  {repr(val):10} -> code {codes[i]} -> category {repr(categories[codes[i]])}")

print()
if codes[0] == codes[1] == codes[2]:
    print("❌ BUG ROOT CAUSE: factorize() treats '', '\\x00', and '\\x00\\x00' as identical!")
    print("   All three distinct strings are mapped to the same category.")

# Test case 3: Impact demonstration
print("\n" + "=" * 60)
print("Test 3: Impact on Data Operations")
print("=" * 60)

# Show how this could affect real-world operations
distinct_values = ['', '\x00', 'a', 'b']
arr = np.array(distinct_values, dtype=object)

print(f"Distinct values: {repr(distinct_values)}")
print(f"Expected: 4 unique hash values")

hashes_categorized = hash_array(arr, categorize=True)
unique_hashes = len(set(hashes_categorized))

print(f"Actual unique hashes with categorize=True: {unique_hashes}")
print(f"Hash values: {hashes_categorized}")

if unique_hashes < len(distinct_values):
    print(f"❌ DATA INTEGRITY ISSUE: Only {unique_hashes} unique hashes for {len(distinct_values)} distinct values!")
    print("   This could cause incorrect groupby, deduplication, or equality checks.")
```

<details>

<summary>
Output demonstrating hash collision and data integrity issue
</summary>
```
/home/npc/pbt/agentic-pbt/worker_/46/repo.py:40: FutureWarning: factorize with argument that is not not a Series, Index, ExtensionArray, or np.ndarray is deprecated and will raise in a future version.
  codes, categories = factorize(test_data)
============================================================
Test 1: Basic Hash Collision with Empty String and Null Byte
============================================================
Input data: ['', '\x00']
Input array dtype: object

Hash with categorize=True:  [1760245841805064774 1760245841805064774]
Hash with categorize=False: [1760245841805064774 7984136654223058057]

❌ BUG CONFIRMED: Empty string '' and null byte '\x00' have the SAME hash when categorize=True
✓ Correct: Empty string '' and null byte '\x00' have DIFFERENT hashes when categorize=False

============================================================
Test 2: Root Cause - factorize() Behavior
============================================================
Input strings: ['', '\x00', '\x00\x00', 'a', '\x00b']
Factorize codes: [0 0 0 1 0]
Unique categories: ['', 'a']

Mapping:
  ''         -> code 0 -> category ''
  '\x00'     -> code 0 -> category ''
  '\x00\x00' -> code 0 -> category ''
  'a'        -> code 1 -> category 'a'
  '\x00b'    -> code 0 -> category ''

❌ BUG ROOT CAUSE: factorize() treats '', '\x00', and '\x00\x00' as identical!
   All three distinct strings are mapped to the same category.

============================================================
Test 3: Impact on Data Operations
============================================================
Distinct values: ['', '\x00', 'a', 'b']
Expected: 4 unique hash values
Actual unique hashes with categorize=True: 3
Hash values: [ 1760245841805064774  1760245841805064774 13950350942979735504
 12688059582079114975]
❌ DATA INTEGRITY ISSUE: Only 3 unique hashes for 4 distinct values!
   This could cause incorrect groupby, deduplication, or equality checks.
```
</details>

## Why This Is A Bug

This violates multiple fundamental expectations and documented behaviors:

1. **Hash Function Contract Violation**: Hash functions must maintain the property that different inputs produce different outputs (with high probability). The empty string `''` and null byte `'\x00'` are distinct Python strings but incorrectly hash to the same value (1760245841805064774) when `categorize=True`.

2. **Documentation Inconsistency**: The `categorize` parameter is documented in pandas/core/util/hashing.py:249-251 as "Whether to first categorize object arrays before hashing. This is more efficient when the array contains duplicate values." The documentation presents this as a performance optimization, not a behavior change. Users reasonably expect an optimization parameter to maintain correctness while improving performance.

3. **Silent Data Corruption**: This bug causes silent failures in core pandas operations:
   - `groupby()` would incorrectly group empty strings with null-byte strings
   - `drop_duplicates()` would incorrectly consider them duplicates
   - `merge()` operations would produce incorrect joins
   - Any equality checking based on hashes would be wrong

4. **Root Cause in factorize()**: The bug originates in pandas' `factorize()` function which incorrectly treats any string containing null bytes as equivalent to an empty string. This affects not just single null bytes but also strings like `'\x00\x00'` and `'\x00b'`.

5. **Inconsistent Behavior**: The same `hash_array` function produces correct results when `categorize=False`, proving the data can be correctly distinguished. The optimization path breaks correctness.

## Relevant Context

- **Affected pandas versions**: Confirmed in pandas 2.3.2
- **Python version**: Tested on Python 3.13
- **Real-world impact**: Null bytes commonly appear in:
  - Binary file parsing
  - Network protocol data
  - Database BLOB fields
  - C-style string termination handling
  - Cryptographic operations

The bug is particularly insidious because:
- It only manifests with specific data patterns
- No warnings or errors are raised
- The incorrect behavior looks plausible (same hash for "similar looking" empty values)
- It affects a performance optimization that may be enabled by default in many pandas operations

Related pandas documentation:
- hash_array: https://pandas.pydata.org/docs/reference/api/pandas.core.util.hashing.hash_array.html
- factorize: https://pandas.pydata.org/docs/reference/api/pandas.factorize.html

## Proposed Fix

The fix requires correcting the `factorize()` function to properly distinguish between empty strings and strings containing null bytes. Based on the code analysis, the issue appears to be in how factorize processes object arrays. Here's the conceptual fix:

```diff
# In pandas factorize function or its underlying implementation:
# The current implementation appears to be treating null bytes as string terminators
# This needs to be changed to properly handle null bytes as valid string content

- # Current behavior (conceptual - exact location needs investigation):
- if '\x00' in string:
-     treat_as_empty_string()
+ # Fixed behavior:
+ # Treat each unique byte sequence as a distinct value
+ # Don't interpret null bytes as terminators in Python strings
```

A workaround for users until this is fixed:
1. Use `categorize=False` when dealing with data that may contain null bytes
2. Pre-process data to escape null bytes before hashing
3. Use alternative hashing methods for binary data

The proper fix requires deeper investigation into the factorize implementation, likely in the Cython/C code that handles string factorization, where C-style null termination may be incorrectly applied to Python strings.