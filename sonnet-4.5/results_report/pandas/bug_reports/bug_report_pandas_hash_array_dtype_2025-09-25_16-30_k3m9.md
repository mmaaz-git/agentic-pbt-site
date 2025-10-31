# Bug Report: pandas.core.util.hashing Inconsistent Hashing Across Integer Dtypes for Negative Values

**Target**: `pandas.core.util.hashing.hash_array` and `pandas.core.util.hashing.hash_pandas_object`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The hashing functions produce different hash values for the same negative integer depending on the dtype (int32 vs int64), while positive integers hash consistently across dtypes. This inconsistent behavior violates the principle of value-based hashing and can cause subtle bugs in deduplication, grouping, and comparison operations.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from pandas.core.util.hashing import hash_array

@given(st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=50))
def test_hash_array_consistent_across_dtypes(values):
    arr_int32 = np.array(values, dtype=np.int32)
    arr_int64 = np.array(values, dtype=np.int64)

    hash_int32 = hash_array(arr_int32)
    hash_int64 = hash_array(arr_int64)

    assert np.array_equal(hash_int32, hash_int64)
```

**Failing input**: `values=[-1]`

## Reproducing the Bug

```python
import numpy as np
import pandas as pd
from pandas.core.util.hashing import hash_array, hash_pandas_object

arr_int32 = np.array([-1], dtype=np.int32)
arr_int64 = np.array([-1], dtype=np.int64)

hash_int32 = hash_array(arr_int32)[0]
hash_int64 = hash_array(arr_int64)[0]

print(f"Hash of -1 as int32: {hash_int32}")
print(f"Hash of -1 as int64: {hash_int64}")
print(f"Equal: {hash_int32 == hash_int64}")

s_int32 = pd.Series([-1, -2, -10], dtype='int32')
s_int64 = pd.Series([-1, -2, -10], dtype='int64')

hash_s32 = hash_pandas_object(s_int32, index=False)
hash_s64 = hash_pandas_object(s_int64, index=False)

print(f"\nSeries hashes equal: {hash_s32.equals(hash_s64)}")

arr_pos_int32 = np.array([1], dtype=np.int32)
arr_pos_int64 = np.array([1], dtype=np.int64)

print(f"\nPositive values work correctly:")
print(f"Equal: {hash_array(arr_pos_int32)[0] == hash_array(arr_pos_int64)[0]}")
```

## Why This Is A Bug

1. **Inconsistent behavior**: Negative integers hash differently across dtypes, but positive integers hash consistently. This asymmetry is unexpected and undocumented.

2. **Violates value semantics**: The value `-1` should produce the same hash regardless of whether it's stored as int32 or int64. Users expect hash functions to be value-based, not representation-based.

3. **Real-world impact**: This can cause bugs in:
   - Deduplication operations where data with different dtypes should be treated as identical
   - Grouping operations that rely on hash values
   - Comparison operations using hashed values

4. **Undocumented**: The docstring for `hash_array` states "Given a 1d array, return an array of deterministic integers" with no mention of dtype-dependent behavior.

## Fix

The root cause is that the hash function is hashing the raw byte representation of values. For negative integers, the two's complement representation differs in byte length between int32 (4 bytes) and int64 (8 bytes):
- int32: `-1` = `0xFFFFFFFF` (4 bytes)
- int64: `-1` = `0xFFFFFFFFFFFFFFFF` (8 bytes)

The fix should normalize integer values to a canonical representation (e.g., int64 or Python int) before hashing, or explicitly document that dtype affects the hash and this is intentional behavior. Given that positive values already hash consistently, the best fix is to ensure negative values do too.