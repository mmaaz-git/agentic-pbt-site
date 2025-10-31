# Bug Report: pandas.api.types.union_categoricals Missing Categories

**Target**: `pandas.api.types.union_categoricals`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`union_categoricals` silently drops categories and converts values to NaN when combining 3+ categoricals with specific overlapping category patterns. Categories are lost when the first categorical contains category 'X', the second contains both 'X' and 'Y\x00' (with null byte), and the third contains 'Y'.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.api.types import union_categoricals
import pandas as pd

categorical_strategy = st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=20).map(
    lambda x: pd.Categorical(x)
)

@given(st.lists(categorical_strategy, min_size=1, max_size=5))
@settings(max_examples=500)
def test_union_categoricals_preserves_all_categories(categoricals):
    result = union_categoricals(categoricals)

    all_input_categories = set()
    for cat in categoricals:
        all_input_categories.update(cat.categories)

    result_categories = set(result.categories)

    assert all_input_categories == result_categories, \
        f"Categories mismatch. Input: {all_input_categories}, Result: {result_categories}"
```

**Failing input**: `categoricals=[pd.Categorical(['0']), pd.Categorical(['0', '1\x00']), pd.Categorical(['1'])]`

## Reproducing the Bug

```python
import pandas as pd
from pandas.api.types import union_categoricals

cat_a = pd.Categorical(['a'])
cat_b = pd.Categorical(['a', 'b\x00'])
cat_c = pd.Categorical(['b'])

result = union_categoricals([cat_a, cat_b, cat_c])

print(f"Expected categories: ['a', 'b\\x00', 'b']")
print(f"Actual categories: {result.categories.tolist()}")
print(f"Expected values: ['a', 'a', 'b\\x00', 'b']")
print(f"Actual values: {result.tolist()}")
```

Output:
```
Expected categories: ['a', 'b\x00', 'b']
Actual categories: ['a', 'b\x00']
Expected values: ['a', 'a', 'b\x00', 'b']
Actual values: ['a', 'a', 'b\x00', nan]
```

## Why This Is A Bug

The docstring for `union_categoricals` states it will "Combine list-like of Categorical-like, unioning categories." The function should preserve all unique categories from all input categoricals. However, it silently drops the category 'b' and converts the corresponding value to NaN, causing data loss.

This violates the documented behavior and causes silent data corruption. Interestingly, when combining just the second and third categoricals (without the first), it works correctly, indicating a flaw in the merging logic when there are 3+ categoricals with overlapping categories.

## Fix

Without access to the source code, the fix likely involves ensuring the category merging logic correctly handles all unique categories regardless of:
1. The number of input categoricals
2. The order of categories
3. The presence of similar category names (e.g., 'b' vs 'b\x00')

The bug appears to be in how pandas tracks which categories have already been added to the union when processing multiple categoricals with overlapping category sets.