# Bug Report: pandas.core.util.hashing.hash_array Incorrect Hash Collision

**Target**: `pandas.core.util.hashing.hash_array`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `hash_array` function produces incorrect hash collisions when `categorize=True`, causing distinct string values (empty string `''` and null byte `'\x00'`) to hash to the same value. This violates the fundamental property that different values should have different hashes.

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
```

**Failing input**: `data=['', '\x00']`

## Reproducing the Bug

```python
import numpy as np
from pandas.core.util.hashing import hash_array

data = ['', '\x00']
arr = np.array(data, dtype=object)

hash_with_categorize = hash_array(arr, categorize=True)
hash_without_categorize = hash_array(arr, categorize=False)

print("Input:", data)
print("Hash with categorize=True: ", hash_with_categorize)
print("Hash with categorize=False:", hash_without_categorize)

assert hash_with_categorize[0] == hash_with_categorize[1], \
    "BUG: Empty string and null byte have SAME hash with categorize=True"

assert hash_without_categorize[0] != hash_without_categorize[1], \
    "With categorize=False, they correctly have DIFFERENT hashes"
```

Output:
```
Input: ['', '\x00']
Hash with categorize=True:  [1760245841805064774 1760245841805064774]
Hash with categorize=False: [1760245841805064774 7984136654223058057]
```

## Why This Is A Bug

1. **Hash collision**: The empty string `''` and null byte `'\x00'` are distinct strings but hash to the same value (1760245841805064774) when `categorize=True`. This is incorrect.

2. **Inconsistent behavior**: The `categorize` parameter is documented as a performance optimization ("more efficient when the array contains duplicate values"), but it changes the hash output, which violates the expected behavior of an optimization parameter.

3. **Data integrity**: Hash functions must maintain the property that different inputs produce different hashes (to avoid collisions). This bug breaks that guarantee for certain string inputs.

4. **High severity**: This can lead to incorrect results in pandas operations that rely on hashing for equality checks, grouping, or deduplication.

## Fix

The root cause appears to be in how the categorization path processes string data. When `categorize=True`, the function calls `factorize()` and then hashes via the Categorical code path, which somehow treats the empty string and null byte as equivalent.

A proper fix would require:
1. Investigating the `factorize()` function or the Categorical hashing to identify where the collision occurs
2. Ensuring that the categorical hashing preserves distinctions between all unique values
3. Adding regression tests to prevent this issue in the future

Recommended investigation starting point: `pandas.core.util.hashing._hash_ndarray` lines 311-323 where the categorize path is implemented.