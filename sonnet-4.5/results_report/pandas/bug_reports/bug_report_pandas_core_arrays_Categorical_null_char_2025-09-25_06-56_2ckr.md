# Bug Report: pandas.core.arrays.Categorical Null Character Data Corruption

**Target**: `pandas.core.arrays.Categorical`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Categorical silently corrupts data by treating the null character `'\x00'` as equivalent to the empty string `''`, causing distinct string values to be merged into a single category and resulting in data loss.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas.core.arrays as arrays

@given(st.lists(st.one_of(st.text(min_size=0, max_size=5), st.none()), min_size=1, max_size=30))
@settings(max_examples=500)
def test_categorical_codes_bounds(values):
    cat = arrays.Categorical(values)

    assert len(cat) == len(values)
    assert len(cat.codes) == len(values)

    for i, code in enumerate(cat.codes):
        if values[i] is None:
            assert code == -1, f"NA should have code -1 at index {i}, got {code}"
        else:
            assert 0 <= code < len(cat.categories), f"Code {code} out of bounds at index {i}"
            assert cat.categories[code] == values[i], f"Code mapping wrong at {i}"
```

**Failing input**: `values=['', '\x00']`

## Reproducing the Bug

```python
import pandas.core.arrays as arrays

values = ['', '\x00']
cat = arrays.Categorical(values)

print(f"Input: {repr(values)}")
print(f"Categories: {repr(list(cat.categories))}")
print(f"Codes: {list(cat.codes)}")

reconstructed = [cat.categories[c] if c != -1 else None for c in cat.codes]
print(f"Original:      {repr(values)}")
print(f"Reconstructed: {repr(reconstructed)}")
print(f"Data preserved: {values == reconstructed}")
```

**Output:**
```
Input: ['', '\x00']
Categories: ['']
Codes: [0, 0]
Original:      ['', '\x00']
Reconstructed: ['', '']
Data preserved: False
```

## Why This Is A Bug

The null character `'\x00'` is a valid, distinct character in Python strings. It has length 1 and is not equal to the empty string `''` (length 0). When creating a Categorical from `['', '\x00']`, pandas should create two distinct categories, not merge them.

This violates the fundamental property of Categorical: distinct input values should map to distinct categories (unless explicitly specified otherwise). The current behavior causes silent data corruption - retrieving values via `cat[i]` or reconstructing from `cat.categories[cat.codes]` will lose the null character and replace it with an empty string.

This can corrupt data in real-world scenarios where:
- String data contains embedded null characters (e.g., from C strings, binary data)
- Users need to distinguish between empty strings and null-terminated strings
- Data roundtripping through Categorical must preserve all values exactly

## Fix

The bug likely stems from pandas using string comparison or hashing that treats `'\x00'` specially, possibly due to C string handling conventions where `'\x00'` terminates strings. The fix would require ensuring that pandas' internal category deduplication and lookup logic treats `'\x00'` as a distinct character, not as a string terminator or equivalent to `''`.

A potential fix would involve reviewing the Categorical constructor and category inference logic to ensure it uses Python's native string equality rather than any C-string-based comparison that might treat `'\x00'` specially.