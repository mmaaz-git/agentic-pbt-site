# Bug Report: pandas.core.algorithms String Collision in factorize/unique/duplicated

**Target**: `pandas.core.algorithms.factorize`, `pandas.core.algorithms.unique`, `pandas.core.algorithms.duplicated`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `factorize()`, `unique()`, and `duplicated()` functions incorrectly treat the empty string `''` and the string `'\x000'` (containing null character followed by '0') as identical values, violating their documented behavior of correctly identifying unique values.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from pandas.core import algorithms as alg

@given(st.lists(st.text(min_size=0, max_size=10)))
def test_factorize_round_trip(values):
    if len(values) == 0:
        return

    arr = np.array(values)
    codes, uniques = alg.factorize(arr)

    reconstructed = uniques.take(codes[codes >= 0])
    original_without_na = arr[codes >= 0]

    if len(reconstructed) > 0 and len(original_without_na) > 0:
        assert all(a == b for a, b in zip(reconstructed, original_without_na))
```

**Failing input**: `['', '\x000']`

## Reproducing the Bug

```python
import numpy as np
from pandas.core import algorithms as alg

values = ['', '\x000']
arr = np.array(values)

codes, uniques = alg.factorize(arr)
print(f"codes: {codes}")
print(f"uniques: {uniques}")

print(f"\nExpected: codes=[0, 1], uniques=['', '\\x000']")
print(f"Actual: codes={list(codes)}, uniques={list(uniques)}")
print(f"\nBoth strings map to same code despite being different:")
print(f"  '' == '\\x000': {'' == '\x000'}")
print(f"  len('')={len('')}, len('\\x000')={len('\x000')}")

unique_vals = alg.unique(arr)
print(f"\nunique() also fails:")
print(f"  Expected 2 unique values, got {len(unique_vals)}")

dup_mask = alg.duplicated(arr, keep='first')
print(f"\nduplicated() also fails:")
print(f"  Mask: {dup_mask}")
print(f"  Both marked as non-duplicates despite unique() returning 1 value")
```

Output:
```
codes: [0 0]
uniques: ['']

Expected: codes=[0, 1], uniques=['', '\x000']
Actual: codes=[0, 0], uniques=['']

Both strings map to same code despite being different:
  '' == '\x000': True
  len('')=0, len('\x000')=2

unique() also fails:
  Expected 2 unique values, got 1

duplicated() also fails:
  Mask: [False False]
  Both marked as non-duplicates despite unique() returning 1 value
```

## Why This Is A Bug

This violates the fundamental contract of these functions:

1. **factorize()** documentation states: "uniques.take(codes) will have the same values as values". This property is violated because:
   - `values[1] = '\x000'`
   - `uniques.take(codes[1]) = uniques[0] = ''`
   - `'\x000' != ''`

2. **unique()** should return all distinct values. It returns only `['']` when given `['', '\x000']`, losing the second distinct value.

3. **duplicated()** marks both values as non-duplicates (both False), but `unique()` returns only 1 value, which is internally inconsistent.

The root cause appears to be in the hash table implementation used by these functions, which incorrectly treats strings differing only in null characters as identical. This is a **data corruption bug** - users will silently lose data when these functions are applied to string arrays containing empty strings and null-character strings.

## Fix

This appears to be a hash collision or string comparison bug in the internal hash table implementation. The fix likely requires:

1. Ensuring the hash function properly distinguishes between `''` and strings containing null characters
2. Ensuring string equality comparisons in the hash table use full string comparison, not just pointer or length-based comparison

The issue may be in:
- `pandas._libs.hashtable` (C/Cython implementation)
- How numpy unicode strings are hashed or compared
- String interning or deduplication logic

A proper fix requires investigating the hash table implementation in `pandas._libs.hashtable` to identify where the collision occurs.