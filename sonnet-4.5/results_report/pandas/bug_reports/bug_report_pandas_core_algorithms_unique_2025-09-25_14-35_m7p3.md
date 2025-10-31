# Bug Report: pandas.core.algorithms.unique Null Character String Collision

**Target**: `pandas.core.algorithms.unique`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `unique` function incorrectly treats strings containing only null characters (`\x00`) as identical to the empty string or other null-character-only strings, failing to return all distinct values. This bug is related to the same underlying issue affecting `factorize` (see separate bug report).

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from pandas.core.algorithms import unique

@given(st.lists(st.text(min_size=0, max_size=10)))
@settings(max_examples=1000)
def test_unique_returns_all_distinct_values(values):
    values_array = np.array(values, dtype=object)
    uniques = unique(values_array)

    assert len(uniques) == len(set(values))
    assert set(uniques) == set(values)
```

**Failing input**: `values=['', '\x00']`

## Reproducing the Bug

```python
import numpy as np
from pandas.core.algorithms import unique

values = np.array(['', '\x00'], dtype=object)
uniques = unique(values)

print(f"Input: {[repr(v) for v in values]}")
print(f"Uniques: {[repr(v) for v in uniques]}")
print(f"Expected: 2 unique values, Got: {len(uniques)}")
```

**Output:**
```
Input: ["''", "'\\x00'"]
Uniques: ["''"]
Expected: 2 unique values, Got: 1
```

**Expected behavior:** The two distinct strings `''` (empty) and `'\x00'` (null character) should both be returned by `unique()`.

**Additional failing cases:**
- `['\x00', '\x00\x00']` - returns only one value instead of two
- `['', '\x00', 'a']` - returns only `['', 'a']` instead of `['', '\x00', 'a']`

## Why This Is A Bug

1. **Violates fundamental property**: `unique()` should return all distinct values from the input. Python correctly identifies `'' != '\x00'`, but `unique()` treats them as the same.

2. **Data loss**: Users calling `unique()` on data with null characters will silently lose distinct values, leading to incorrect analysis and statistics.

3. **Affects realistic use cases**: Same as factorize bug - null characters legitimately appear in binary data, C-string imports, text with control characters, and database values.

4. **Widespread impact**: This bug affects a fundamental building block used throughout pandas for deduplication, grouping, and analysis.

## Relationship to factorize Bug

This bug shares the same root cause as the `factorize()` bug (see separate report). Both functions likely use the same underlying hash table or string comparison code that incorrectly treats `\x00` as a string terminator.

Evidence:
```python
values = np.array(['', '\x00'], dtype=object)
print(unique(values))          # ["''" ]
codes, uniques = factorize(values)
print(uniques)                 # ["''"]
```

Both functions return the same incorrect result, confirming a shared underlying issue.

## Fix

The fix is the same as for `factorize`: the underlying hash table or string comparison logic in pandas.core.algorithms needs to properly handle strings containing null bytes, treating them as valid characters rather than terminators. This likely requires fixing code in:
- Hash table implementation for object dtype arrays
- String comparison/equality functions used in factorization and unique value detection
- Possibly in Cython/C-level code that interfaces with Python strings