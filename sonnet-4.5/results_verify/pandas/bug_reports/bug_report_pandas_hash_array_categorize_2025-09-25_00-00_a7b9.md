# Bug Report: pandas.core.util.hashing.hash_array Categorize Parameter Violates Documented Contract

**Target**: `pandas.core.util.hashing.hash_array`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `categorize` parameter in `hash_array()` is documented as an optimization that should not affect the hash result, but it produces different hashes for certain inputs like `['', '\x00']` (empty string and null byte).

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
from pandas.core.util.hashing import hash_array

@given(st.lists(st.text(min_size=0, max_size=100), min_size=1))
@settings(max_examples=500)
def test_hash_array_categorize_consistency(values):
    arr = np.array(values, dtype=object)
    hash_with_categorize = hash_array(arr, categorize=True)
    hash_without_categorize = hash_array(arr, categorize=False)
    assert np.array_equal(hash_with_categorize, hash_without_categorize), \
        "categorize parameter should be optimization only, not change result"
```

**Failing input**: `values=['', '\x00']`

## Reproducing the Bug

```python
import numpy as np
from pandas.core.util.hashing import hash_array

values = ['', '\x00']
arr = np.array(values, dtype=object)

hash_with_categorize = hash_array(arr, categorize=True)
hash_without_categorize = hash_array(arr, categorize=False)

print("Hash with categorize=True: ", hash_with_categorize)
print("Hash with categorize=False:", hash_without_categorize)
print("Equal?", np.array_equal(hash_with_categorize, hash_without_categorize))
```

Output:
```
Hash with categorize=True:  [1760245841805064774 1760245841805064774]
Hash with categorize=False: [1760245841805064774 7984136654223058057]
Equal? False
```

## Why This Is A Bug

The documentation for the `categorize` parameter states (line 249 in hashing.py):

> "Whether to first categorize object arrays before hashing. This is more efficient when the array contains duplicate values."

This clearly describes `categorize` as a performance optimization, implying it should not change the semantic result. However, when `categorize=True`, the function uses `pandas.factorize()` which incorrectly treats the empty string `''` and null byte `'\x00'` as the same category, causing both values to receive identical hash codes `[0, 0]`.

This violates the contract that the `categorize` parameter should only affect performance, not correctness.

## Fix

The root cause is in `pandas.factorize()` treating `''` and `'\x00'` as identical when they are distinct strings. A potential fix would be to investigate why factorize conflates these values and ensure it properly distinguishes between all unique string values.

Alternatively, the documentation should be updated to warn that `categorize=True` may produce different (and potentially incorrect) hashes for certain edge cases, though this would be admitting the optimization is unsound.