# Bug Report: pandas Index.unique() Null Byte Deduplication

**Target**: `pandas.Index.unique()`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When an Index contains both empty string `''` and null byte `'\x00'`, AND contains duplicate values of any kind, `.unique()` incorrectly deduplicates `'\x00'` as if it were `''`, causing silent data loss.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd
from pandas.api.types import union_categoricals


@settings(max_examples=1000)
@given(
    st.lists(st.text(), min_size=1, max_size=50),
    st.lists(st.text(), min_size=1, max_size=50),
)
def test_union_categoricals_preserves_all_values(vals1, vals2):
    cat1 = pd.Categorical(vals1)
    cat2 = pd.Categorical(vals2)

    result = union_categoricals([cat1, cat2])
    result_list = list(result)

    expected = list(cat1) + list(cat2)

    assert result_list == expected
```

**Failing input**: `vals1=['', '0'], vals2=['0', '\x00']`

## Reproducing the Bug

```python
import pandas as pd

idx = pd.Index(['', 'x', '\x00', 'x'])
unique_vals = idx.unique()

print("Original:", list(idx))
print("Unique:", list(unique_vals))

assert '' in unique_vals
assert '\x00' in unique_vals
assert len(unique_vals) == 3
```

Output:
```
Original: ['', 'x', '\x00', 'x']
Unique: ['', 'x']
AssertionError: '\x00' not in Index(['', 'x'], dtype='object')
```

**Trigger conditions:**
- Index must contain both `''` and `'\x00'`
- Index must contain duplicates of ANY value (including unrelated values)
- Without duplicates, `unique()` works correctly: `pd.Index(['', '\x00']).unique()` → `['', '\x00']` ✓

## Why This Is A Bug

The null byte `'\x00'` and empty string `''` are distinct Python strings:
- `'' != '\x00'` is True
- `len('')` is 0, `len('\x00')` is 1
- They have different hash values

Yet `Index.unique()` incorrectly treats them as duplicates when the index contains any duplicate values, causing silent data corruption.

This affects real use cases including:
- `union_categoricals()` which uses `Index.unique()` to merge categories (values map to NaN when their category is lost)
- Any code deduplicating string indices containing both empty strings and null bytes
- Data processing pipelines that use pandas for string data with null terminators

## Fix

The issue likely stems from how pandas internally handles string comparison or hashing for the null byte character. The fix would require investigating the C implementation of Index.unique() or the underlying hash table implementation to ensure null bytes are treated as distinct from empty strings.

Without access to the exact implementation details, the general approach would be:
1. Identify where in the unique() implementation strings are being compared or hashed
2. Ensure that null byte characters are handled correctly and not confused with empty strings
3. Add tests to prevent regression