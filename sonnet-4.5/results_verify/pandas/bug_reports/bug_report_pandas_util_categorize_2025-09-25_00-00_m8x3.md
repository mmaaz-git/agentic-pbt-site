# Bug Report: pandas.util.hash_array - categorize Parameter Produces Incorrect Hashes

**Target**: `pandas.util.hash_array`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `hash_array` function with `categorize=True` produces incorrect hash values for certain inputs containing duplicate values and distinct characters like empty strings and null bytes. Different input values incorrectly receive the same hash, violating the fundamental property of hash functions.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, assume
import pandas.util

@given(st.lists(st.text(), min_size=2, max_size=50))
def test_hash_array_categorize_parameter(lst):
    assume(len(set(lst)) < len(lst))
    arr = np.array(lst, dtype=object)
    hash_with_categorize = pandas.util.hash_array(arr, categorize=True)
    hash_without_categorize = pandas.util.hash_array(arr, categorize=False)

    assert np.array_equal(hash_with_categorize, hash_without_categorize)
```

**Failing input**: `lst=['', '', '\x00']`

## Reproducing the Bug

```python
import numpy as np
import pandas.util

arr = np.array(['', '', '\x00'], dtype=object)

hash_cat = pandas.util.hash_array(arr, categorize=True)
hash_no_cat = pandas.util.hash_array(arr, categorize=False)

print(f"categorize=True:  {hash_cat}")
print(f"categorize=False: {hash_no_cat}")
```

Output:
```
categorize=True:  [1760245841805064774 1760245841805064774 1760245841805064774]
categorize=False: [1760245841805064774 1760245841805064774 7984136654223058057]
```

Notice that with `categorize=True`, all three values (including the distinct '\x00' character) receive the same hash, which is incorrect.

## Why This Is A Bug

1. **Hash function property violated**: A fundamental property of hash functions is that different values should (with high probability) have different hashes. Here, '' and '\x00' are distinct values but receive identical hashes when `categorize=True`.

2. **Documentation claims it's an optimization**: The parameter is documented as:
   ```
   categorize : bool, default True
       Whether to first categorize object arrays before hashing. This is more
       efficient when the array contains duplicate values.
   ```

   The phrase "more efficient" implies this is a performance optimization that should produce the same results. If it produces different results, it's not an optimization - it's a different algorithm.

3. **Breaks consistency**: Code relying on `hash_array` for deduplication, integrity checking, or hash-based comparisons will get incorrect results when using the default `categorize=True`.

4. **With `categorize=False`, the hashes are correct**: The same input produces the expected behavior where '' and '' get the same hash, but '\x00' gets a different hash.

## Impact

- **Severity: High** because:
  - This affects the default behavior (`categorize=True` is the default)
  - It silently produces incorrect hashes leading to data corruption in downstream applications
  - It violates core hash function properties
  - Real-world data often contains empty strings and special characters

## Fix

The root cause appears to be in how the categorization path handles object arrays. When `categorize=True`, the function:
1. Calls `factorize()` to create categories
2. Creates a Categorical
3. Hashes the categorical representation

The bug likely occurs in how the Categorical is hashed - it may be using codes instead of the actual category values, or there may be an issue in how `factorize()` handles empty strings vs null bytes.

A proper fix would require:
1. Investigating the `_hash_pandas_object` method on Categorical objects
2. Ensuring that the categorical hashing path produces identical results to the non-categorical path
3. Adding regression tests for edge cases with empty strings, null bytes, and other special characters
4. Potentially reviewing the use of `factorize()` in this context

**Temporary workaround**: Users should explicitly set `categorize=False` when hashing object arrays containing special characters until this is fixed.