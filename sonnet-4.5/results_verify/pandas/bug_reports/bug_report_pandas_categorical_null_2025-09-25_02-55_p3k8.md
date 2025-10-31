# Bug Report: pandas.Categorical Null Character Handling

**Target**: `pandas.Categorical`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`pd.Categorical` treats strings that differ only by trailing null characters (`\x00`) as identical, causing silent data corruption. When creating a Categorical from `['', '\x00']` or `['a', 'a\x00']`, both values are assigned to the same category, and the null character is lost during reconstruction.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd
from hypothesis import given, strategies as st, settings


@given(arr=st.lists(st.text(min_size=0, max_size=10), min_size=1, max_size=50))
@settings(max_examples=500)
def test_categorical_round_trip(arr):
    cat = pd.Categorical(arr)
    reconstructed = cat.categories[cat.codes]

    assert list(reconstructed) == arr, \
        f"Categorical round-trip failed: original {arr}, reconstructed {list(reconstructed)}"
```

**Failing input**: `['', '\x00']`

## Reproducing the Bug

```python
import pandas as pd

arr = ['', '\x00']
cat = pd.Categorical(arr)

print(f"Input:      {repr(arr)}")
print(f"Categories: {repr(list(cat.categories))}")
print(f"Codes:      {cat.codes.tolist()}")

reconstructed = list(cat.categories[cat.codes])
print(f"Output:     {repr(reconstructed)}")
```

Output:
```
Input:      ['', '\x00']
Categories: ['']
Codes:      [0, 0]
Output:     ['', '']
```

Additional examples:
```python
pd.Categorical(['a', 'a\x00'])
```
Produces only one category `['a']` with codes `[0, 0]`, losing the null character.

However, this works correctly:
```python
pd.Categorical(['\x00', '\x01'])
```
Produces two categories as expected.

## Why This Is A Bug

The null character `\x00` is a valid string character in Python that should be preserved. Pandas Categorical is incorrectly treating strings that differ only by trailing null characters as identical, which causes silent data corruption.

This violates:
- The fundamental expectation that distinct input values remain distinct
- The round-trip property: `Categorical(arr)` does not preserve `arr`
- The documented behavior that Categorical represents all distinct values

The bug specifically occurs when:
1. Two strings are identical except one has trailing `\x00` characters
2. Examples: `'' vs '\x00'`, `'a' vs 'a\x00'`
3. But NOT when both strings contain only non-null characters after the null: `'\x00' vs '\x01'` works correctly

## Impact

- Silent data corruption: values are changed without warning
- Data integrity issues in applications handling binary strings or special characters
- Loss of information that could be critical in certain domains (e.g., parsing binary protocols, handling control characters)

## Fix

The issue likely stems from how pandas compares or hashes strings when building the category Index. The fix would involve ensuring that the Index creation properly distinguishes strings with trailing null characters.

Without access to the source, the probable locations are:
1. `pandas.Categorical.__init__`: Category uniquification logic
2. `pandas.Index` construction for object dtype: String comparison/hashing

The fix should ensure null characters are treated as significant, not stripped or ignored during category deduplication.