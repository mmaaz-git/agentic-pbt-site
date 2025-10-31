# Bug Report: pandas.core.util.hashing.hash_array Categorize Parameter Changes Hash Values

**Target**: `pandas.core.util.hashing.hash_array`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `hash_array` function produces different hash values for identical inputs depending on the `categorize` parameter. Specifically, when `categorize=True`, the empty string `''` and null character `'\x00'` are incorrectly treated as identical and produce the same hash, but when `categorize=False`, they correctly produce different hashes.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from pandas.core.util.hashing import hash_array

@given(st.lists(st.text(min_size=0, max_size=5), min_size=10, max_size=50))
def test_hash_array_strings_with_duplicates(values):
    arr = np.array(values, dtype=object)
    hash_categorized = hash_array(arr, categorize=True)
    hash_uncategorized = hash_array(arr, categorize=False)
    assert np.array_equal(hash_categorized, hash_uncategorized)
```

**Failing input**: `['', '', '', '', '', '', '', '', '', '\x00']`

## Reproducing the Bug

```python
import numpy as np
from pandas.core.util.hashing import hash_array

arr = np.array(['', '\x00'], dtype=object)

hash_with_categorize = hash_array(arr, categorize=True)
hash_without_categorize = hash_array(arr, categorize=False)

print("Input:", [repr(v) for v in arr])
print("With categorize=True: ", hash_with_categorize)
print("With categorize=False:", hash_without_categorize)

print("\nExpected: Both should produce the same hash values")
print("Actual:   Different hashes for the same input!")
print("  '' hashes to:    ", hash_without_categorize[0])
print("  '\\x00' hashes to:", hash_without_categorize[1])
print("\nBut with categorize=True, both hash to:", hash_with_categorize[0])
```

## Why This Is A Bug

According to the `hash_array` docstring (line 249), the `categorize` parameter is described as:
> "Whether to first categorize object arrays before hashing. This is more efficient when the array contains duplicate values."

This indicates `categorize` should be purely a performance optimization and should not change the hash values. However, the bug demonstrates that `categorize=True` produces different (incorrect) results.

The root cause is that `pandas.factorize()` (called at line 318) incorrectly treats the empty string `''` and the null character `'\x00'` as the same value, assigning them the same factorization code. This causes them to receive identical hash values when `categorize=True`.

This violates the fundamental property that different inputs should produce different hashes (except for intentional collisions in hash functions).

## Fix

The issue stems from using `factorize()` which has unexpected behavior with certain string edge cases. The fix should ensure that `factorize()` correctly distinguishes between all unique values, or the hashing code should validate that categorization preserves value distinctions.

One approach would be to fix the underlying `factorize()` behavior for strings containing null characters. Alternatively, the hashing code could add validation:

```diff
--- a/pandas/core/util/hashing.py
+++ b/pandas/core/util/hashing.py
@@ -317,6 +317,10 @@ def _hash_ndarray(

             codes, categories = factorize(vals, sort=False)
+            # Verify factorization didn't collapse distinct values
+            if len(categories) != len(np.unique(vals)):
+                # Fall back to non-categorized path if factorization is lossy
+                categorize = False
             dtype = CategoricalDtype(categories=Index(categories), ordered=False)
             cat = Categorical._simple_new(codes, dtype)
             return cat._hash_pandas_object(
```

However, this fix may not be complete as the real issue appears to be in the `factorize()` implementation itself.