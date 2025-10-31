# Bug Report: pandas.core.groupby Categorical Ordering with sort=False

**Target**: `pandas.core.groupby.DataFrameGroupBy` / `pandas.core.groupby.SeriesGroupBy`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When grouping by a categorical column with `observed=False` and `sort=False`, pandas incorrectly orders the result index by placing observed categories first (in appearance order), followed by unobserved categories. This violates the categorical ordering and creates inconsistent behavior compared to `sort=True`.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, settings, strategies as st

@given(
    st.lists(st.integers(min_value=0, max_value=5), min_size=1, max_size=20),
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100), min_size=1, max_size=20),
)
@settings(max_examples=100)
def test_categorical_groupby_sort_false_with_observed_false(keys, values):
    if len(keys) != len(values):
        return

    categories_ordered = [5, 4, 3, 2, 1, 0]

    df = pd.DataFrame({
        'key': pd.Categorical(keys, categories=categories_ordered, ordered=True),
        'value': values
    })

    result_sort_true = df.groupby('key', observed=False, sort=True).sum()
    result_sort_false = df.groupby('key', observed=False, sort=False).sum()

    expected_order = categories_ordered

    assert list(result_sort_true.index) == expected_order

    assert list(result_sort_false.index) == expected_order, \
        f"sort=False should respect category order when observed=False"
```

**Failing input**: `keys=[0], values=[0.0]`

## Reproducing the Bug

```python
import pandas as pd

df = pd.DataFrame({
    'key': pd.Categorical([2, 0], categories=[5, 4, 3, 2, 1, 0], ordered=True),
    'value': [10, 20]
})

result_sorted = df.groupby('key', observed=False, sort=True).sum()
print("With sort=True:")
print(list(result_sorted.index))

result_unsorted = df.groupby('key', observed=False, sort=False).sum()
print("With sort=False:")
print(list(result_unsorted.index))
```

**Output:**
```
With sort=True:
[5, 4, 3, 2, 1, 0]
With sort=False:
[2, 0, 5, 4, 3, 1]
```

## Why This Is A Bug

Categorical data types have an inherent ordering defined by their categories. When `observed=False` is specified, all categories should be included in the result, and they should respect the categorical order.

The current behavior with `sort=False` is inconsistent:
1. Observed categories [2, 0] appear first in data order
2. Unobserved categories [5, 4, 3, 1] follow

Expected behavior:
- With categorical grouping keys and `observed=False`, the result should always follow the categorical order [5, 4, 3, 2, 1, 0]
- The `sort` parameter should not affect this for categoricals, since their order is already defined by the category order

This violates the principle that categoricals have a pre-defined ordering that should be respected, especially when explicitly requesting all categories with `observed=False`.

## Fix

The issue likely originates in the groupby result index construction when `sort=False`. The fix should ensure that when grouping by a categorical with `observed=False`, the result index follows the categorical order regardless of the `sort` parameter.

A potential fix would be in `pandas/core/groupby/ops.py` or `pandas/core/groupby/grouper.py`, where the result index is constructed. When the grouping key is categorical and `observed=False`, the index should be created from the category order rather than group appearance order.

```diff
# Conceptual fix - actual location may vary
def _get_result_index(self):
    if self.grouping_key_is_categorical and not self.observed:
-       # Current: uses group appearance order when sort=False
-       return index_from_groups(self.groups)
+       # Fixed: always use categorical order for categoricals with observed=False
+       return CategoricalIndex(self.categorical_categories)
    else:
        if self.sort:
            return sorted_index(self.groups)
        else:
            return index_from_groups(self.groups)
```

Note: This is a conceptual patch. The actual fix would need to handle the interaction with the aggregation results and ensure values are aligned correctly with the categorical-ordered index.