# Bug Report: pandas.core.algorithms.unique Null Character Data Corruption

**Target**: `pandas.core.algorithms.unique`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `unique` function incorrectly transforms strings containing null characters (`'\x00'`) into empty strings, causing silent data corruption.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from pandas.core.algorithms import unique

@given(st.lists(st.text(min_size=0, max_size=10), min_size=0, max_size=100))
def test_unique_with_strings(values):
    arr = np.array(values)
    result = unique(arr)
    for val in result:
        assert val in values, f"unique returned value {val} not in input"
```

**Failing input**: `['\x00']`

## Reproducing the Bug

```python
import numpy as np
from pandas.core.algorithms import unique

values = ['\x00']
arr = np.array(values)
result = unique(arr)

print(f"Input: {repr(values)}")
print(f"Result: {repr(result.tolist())}")
print(f"Input[0]: {repr(values[0])} (length: {len(values[0])})")
print(f"Result[0]: {repr(result[0])} (length: {len(result[0])})")
```

Output:
```
Input: ['\x00']
Result: ['']
Input[0]: '\x00' (length: 1)
Result[0]: np.str_('') (length: 0)
```

## Why This Is A Bug

The `unique` function documentation states it "Return[s] unique values based on a hash table" and "Returns unique values in order of appearance". The null character `'\x00'` is a valid Unicode/ASCII character that should be preserved as-is. Instead, `unique` is silently transforming it into an empty string, which is data corruption.

This violates the fundamental property that all values returned by `unique(arr)` should be elements of the input array `arr`.

## Fix

The bug likely occurs during string handling in the underlying C implementation or in numpy string dtype conversion. The issue may be related to C-style string handling where `'\x00'` is treated as a string terminator.

A potential fix would involve:
1. Ensuring proper handling of embedded null characters in the hashing and deduplication logic
2. Using length-aware string comparisons rather than relying on null-termination
3. Preserving the original string data without transformation during the unique operation

This may require changes in `pandas._libs.algos` or the string dtype handling in the `_ensure_data` function that prepares values for the hash table operations.