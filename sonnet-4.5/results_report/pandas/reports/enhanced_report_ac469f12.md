# Bug Report: pandas.api.types.union_categoricals Silently Drops Categories with Null Bytes

**Target**: `pandas.api.types.union_categoricals`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`union_categoricals` silently drops categories and converts values to NaN when combining 3+ categoricals with overlapping categories where one contains a null byte character. The bug manifests when categories differ only by the presence of null bytes (e.g., 'b' vs 'b\x00').

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

<details>

<summary>
**Failing input**: `[pd.Categorical(['0']), pd.Categorical(['0', '1\x00']), pd.Categorical(['1'])]`
</summary>
```
Test failed!
Categories mismatch. Input: {'1', '0', '1\x00'}, Result: {'0', '1\x00'}
```
</details>

## Reproducing the Bug

```python
import pandas as pd
from pandas.api.types import union_categoricals

# Test Case 1: Primary Bug Scenario
cat_a = pd.Categorical(['a'])
cat_b = pd.Categorical(['a', 'b\x00'])
cat_c = pd.Categorical(['b'])

print(f"cat_a: values={cat_a.tolist()}, categories={cat_a.categories.tolist()}")
print(f"cat_b: values={cat_b.tolist()}, categories={cat_b.categories.tolist()}")
print(f"cat_c: values={cat_c.tolist()}, categories={cat_c.categories.tolist()}")

result = union_categoricals([cat_a, cat_b, cat_c])

print(f"Expected categories: ['a', 'b\\x00', 'b']")
print(f"Actual categories: {result.categories.tolist()}")
print(f"Expected values: ['a', 'a', 'b\\x00', 'b']")
print(f"Actual values: {result.tolist()}")
```

<details>

<summary>
Silent data loss: category 'b' dropped, value becomes NaN
</summary>
```
cat_a: values=['a'], categories=['a']
cat_b: values=['a', 'b\x00'], categories=['a', 'b\x00']
cat_c: values=['b'], categories=['b']

Expected categories: ['a', 'b\x00', 'b']
Actual categories: ['a', 'b\x00']
Expected values: ['a', 'a', 'b\x00', 'b']
Actual values: ['a', 'a', 'b\x00', nan]
```
</details>

## Why This Is A Bug

This violates the documented behavior of `union_categoricals` which explicitly states it performs a "union" of categories. By mathematical definition, a union must include ALL unique elements from all input sets. The function's documentation shows examples where all unique categories are preserved when combining categoricals.

The bug causes **silent data corruption** - values that should map to valid categories are converted to NaN without warning. The inconsistent behavior (works with 2 categoricals, fails with 3+) indicates a flaw in the iterative merging logic when processing multiple categoricals.

Critical observations:
1. **Data Loss**: Valid category 'b' is silently dropped from the result
2. **NaN Conversion**: The value 'b' becomes NaN without any error or warning
3. **Order Dependence**: Different input orders produce different missing categories
4. **Inconsistent Behavior**: Combining just cat_b and cat_c works correctly, preserving all three categories ['a', 'b\x00', 'b']

## Relevant Context

The root cause appears to be in how pandas handles the `unique()` operation on Index objects when combining multiple categoricals. Testing reveals:

```python
# This works correctly - direct unique on Index
idx = pd.Index(['a', 'b\x00', 'b'])
idx.unique()  # Returns: ['a', 'b\x00', 'b'] ✓

# This fails - append then unique
idx1 = pd.Index(['a'])
idx2 = pd.Index(['a', 'b\x00'])
idx3 = pd.Index(['b'])
combined = idx1.append([idx2, idx3])  # ['a', 'a', 'b\x00', 'b']
combined.unique()  # Returns: ['a', 'b\x00'] ✗ Missing 'b'!
```

The bug is in the pandas Index implementation where `append().unique()` incorrectly treats strings that differ only by null bytes as duplicates. This affects pandas version 2.3.2.

Documentation reference: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.api.types.union_categoricals.html

## Proposed Fix

The issue is in the `unique()` implementation for pandas Index objects when processing strings with special characters. The bug is triggered by the internal hashtable implementation in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/algorithms.py` which incorrectly handles strings with null bytes.

A high-level fix would involve:
1. Ensuring the hashtable-based unique algorithm in `pandas.core.algorithms.unique_with_mask` correctly distinguishes between strings like 'b' and 'b\x00'
2. Fixing the Index.append().unique() pipeline to preserve all distinct values
3. Adding test cases for union_categoricals with special characters including null bytes

The workaround for users is to avoid using null bytes in categorical data, or to manually reconstruct the categories using numpy's unique function which handles this case correctly.