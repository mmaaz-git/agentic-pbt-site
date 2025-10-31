# Bug Report: pandas.api.types.union_categoricals Associativity Violation and Data Loss

**Target**: `pandas.api.types.union_categoricals`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`union_categoricals` violates associativity and silently converts valid categorical values to NaN when the first categorical has fewer categories than subsequent categoricals.

## Property-Based Test

```python
import pandas as pd
from pandas.api.types import union_categoricals
from hypothesis import given, strategies as st, settings


@settings(max_examples=500)
@given(
    st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=20),
    st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=20),
    st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=20)
)
def test_union_categoricals_three_way_associativity(values1, values2, values3):
    cat1 = pd.Categorical(values1)
    cat2 = pd.Categorical(values2)
    cat3 = pd.Categorical(values3)

    result_left = union_categoricals([union_categoricals([cat1, cat2]), cat3])
    result_right = union_categoricals([cat1, union_categoricals([cat2, cat3])])

    assert list(result_left) == list(result_right)
```

**Failing input**:
- `values1 = ['0']`
- `values2 = ['0']`
- `values3 = ['0\x00']`

## Reproducing the Bug

```python
import pandas as pd
from pandas.api.types import union_categoricals

cat1 = pd.Categorical(['0'], categories=['0'])
cat2 = pd.Categorical(['0'], categories=['0'])
cat3 = pd.Categorical(['0\x00'], categories=['0\x00'])

intermediate2 = union_categoricals([cat2, cat3])
print(f"intermediate2: values={list(intermediate2)}, categories={list(intermediate2.categories)}")

result = union_categoricals([cat1, intermediate2])
print(f"result: values={list(result)}, categories={list(result.categories)}")

print(f"Expected: ['0', '0', '0\\x00']")
print(f"Actual:   {list(result)}")
print(f"Bug: {pd.isna(result[2])}")
```

**Output**:
```
intermediate2: values=['0', '0\x00'], categories=['0', '0\x00']
result: values=['0', '0', nan], categories=['0']
Expected: ['0', '0', '0\x00']
Actual:   ['0', '0', nan]
Bug: True
```

## Why This Is A Bug

The `union_categoricals` function should preserve all values from input categoricals. However, when the first categorical has fewer categories than the second, values in the second categorical that are not in the first's category list are incorrectly converted to NaN. This causes:

1. **Silent data loss**: Valid values become NaN without warning
2. **Associativity violation**: `union_categoricals([a, union_categoricals([b, c])])` ≠ `union_categoricals([union_categoricals([a, b]), c])`
3. **Unexpected behavior**: The result categories list is `['0']` instead of the expected `['0', '0\x00']`

The root cause appears to be in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/dtypes/concat.py:308-314`, where the "fastpath" check incorrectly uses only the first categorical's categories when categories should be unioned.

## Fix

The bug is in the fastpath logic. When `_categories_match_up_to_permutation` returns True, the code assumes categories are identical and uses `first.categories` (line 310). However, this check should verify that all categoricals have exactly the same categories, not just matching dtype hashes.

The fix should either:
1. Make `_categories_match_up_to_permutation` more strict to check for identical category sets, or
2. Remove the fastpath when categories differ, even if they match "up to permutation"

```diff
--- a/pandas/core/dtypes/concat.py
+++ b/pandas/core/dtypes/concat.py
@@ -305,9 +305,12 @@ def union_categoricals(
         raise TypeError("dtype of categories must be the same")

     ordered = False
-    if all(first._categories_match_up_to_permutation(other) for other in to_union[1:]):
-        # identical categories - fastpath
-        categories = first.categories
+    if all(first._categories_match_up_to_permutation(other) for other in to_union[1:]) and \
+       all(set(first.categories) == set(other.categories) for other in to_union[1:]):
+        # identical categories (same set of values) - fastpath
+        # Note: We need to check both hash equality AND set equality because
+        # hash equality doesn't guarantee the categories contain the same values
+        categories = first.categories
         ordered = first.ordered

         all_codes = [first._encode_with_my_categories(x)._codes for x in to_union]
```