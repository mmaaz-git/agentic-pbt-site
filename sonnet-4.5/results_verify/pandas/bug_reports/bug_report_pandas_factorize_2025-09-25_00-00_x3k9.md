# Bug Report: pandas.factorize Null Character Corruption

**Target**: `pandas.factorize` and `pandas.Categorical`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`pd.factorize()` and categorical Series creation silently corrupt null characters (`\x00`) by treating them as equivalent to empty strings (`''`). This violates pandas' documented contract that `uniques.take(codes)` will have the same values as the input, and causes data loss when creating categorical data.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st, settings

@given(st.lists(st.text(), min_size=1, max_size=10))
@settings(max_examples=2000)
def test_factorize_reconstructs_original(values):
    s = pd.Series(values)
    codes, uniques = pd.factorize(s)
    reconstructed = uniques.take(codes)

    for i, (orig, recon) in enumerate(zip(values, reconstructed)):
        if pd.isna(orig) and pd.isna(recon):
            continue
        assert orig == recon, f'Mismatch at index {i}: "{orig}" != "{recon}"'
```

**Failing input**: `['', '\x00']`

## Reproducing the Bug

### Case 1: pd.factorize()

```python
import pandas as pd

values = ['', '\x00']
codes, uniques = pd.factorize(values)
reconstructed = uniques.take(codes)

print('Original:     ', repr(values))
print('Codes:        ', codes)
print('Uniques:      ', repr(uniques.tolist()))
print('Reconstructed:', repr(reconstructed.tolist()))

assert values[0] == reconstructed[0]
assert values[1] == reconstructed[1]
```

**Output:**
```
Original:      ['', '\x00']
Codes:         [0 0]
Uniques:       ['']
Reconstructed: ['', '']
AssertionError: values[1] != reconstructed[1]  # '\x00' != ''
```

### Case 2: Categorical Series

```python
import pandas as pd

values = ['', '\x00']
s = pd.Series(values, dtype='category')

print('Original:  ', repr(values))
print('Categories:', repr(s.cat.categories.tolist()))
print('Retrieved: ', repr([s.iloc[0], s.iloc[1]]))

assert s.iloc[0] == values[0]
assert s.iloc[1] == values[1]
```

**Output:**
```
Original:   ['', '\x00']
Categories: ['']
Retrieved:  ['', '']
AssertionError: s.iloc[1] != values[1]  # '' != '\x00'
```

## Why This Is A Bug

**For pd.factorize()**: The documentation explicitly states: "`uniques.take(codes)` will have the same values as `values`". However, when the input contains both an empty string and a null character, factorize incorrectly treats them as identical.

**For Categorical**: When creating a categorical Series, pandas should preserve all distinct values. Instead, `'\x00'` is corrupted to `''`.

The bug manifests as:
1. Two distinct values (`''` and `'\x00'`) are both assigned the same code `0`
2. Only one unique value/category (`''`) is stored
3. Data retrieval produces `['', '']` instead of `['', '\x00']`

This is **silent data corruption** - users lose information with no warning or error, which can lead to incorrect analysis results.

## Fix

The bug likely stems from C-style string handling where `\x00` is treated as a string terminator. The factorization logic needs to handle null characters as distinct values. The fix would require examining the hashtable implementation in the pandas internals (likely in `pandas/_libs/hashtable.pyx` or similar) to ensure it correctly handles strings containing null bytes when computing uniqueness.