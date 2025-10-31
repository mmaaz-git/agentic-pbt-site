# Bug Report: pandas.core.indexes.api.safe_sort_index Documentation Mismatch

**Target**: `pandas.core.indexes.api.safe_sort_index`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `safe_sort_index` function's docstring claims it "Returns the sorted index" but it may actually return an unsorted index when the elements are not comparable (e.g., mixed int and string types).

## Property-Based Test

```python
from pandas.core.indexes.api import _get_combined_index
from hypothesis import given, strategies as st, settings
import pandas as pd


@st.composite
def index_strategy(draw):
    data = draw(st.one_of(
        st.lists(st.integers(min_value=-100, max_value=100), min_size=0, max_size=15),
        st.lists(st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=5), min_size=0, max_size=15),
    ))
    return pd.Index(data)


@given(idx1=index_strategy(), idx2=index_strategy())
@settings(max_examples=300)
def test_combined_index_sort_actually_sorts(idx1, idx2):
    try:
        result = _get_combined_index([idx1, idx2], intersect=False, sort=True, copy=False)

        if len(result) > 0:
            assert result.is_monotonic_increasing or result.is_monotonic_decreasing, \
                f"sort=True should produce monotonic result: {result}"
    except (TypeError, ValueError) as e:
        assume(False)
```

**Failing input**: `idx1=Index([0], dtype='int64')`, `idx2=Index(['a'], dtype='object')`

## Reproducing the Bug

```python
import pandas as pd
from pandas.core.indexes.api import safe_sort_index

idx = pd.Index([0, 'a'], dtype='object')

result = safe_sort_index(idx)

print("Input:", idx)
print("Result:", result)
print("Is result sorted?", result.is_monotonic_increasing or result.is_monotonic_decreasing)
print()
print("Docstring says:")
print(safe_sort_index.__doc__)
```

**Output**:
```
Input: Index([0, 'a'], dtype='object')
Result: Index([0, 'a'], dtype='object')
Is result sorted? False

Docstring says:

Returns the sorted index

We keep the dtypes and the name attributes.

Parameters
----------
index : an Index

Returns
-------
Index
```

## Why This Is A Bug

The function's docstring states "Returns the sorted index" without any qualification, but the function may return an unsorted index when sorting would fail due to incomparable elements. The actual implementation catches `TypeError` and returns the original unsorted index in such cases.

This violates the documented API contract and can mislead users who expect the function to always return a sorted index.

## Fix

```diff
diff --git a/pandas/core/indexes/api.py b/pandas/core/indexes/api.py
index 1234567..abcdefg 100644
--- a/pandas/core/indexes/api.py
+++ b/pandas/core/indexes/api.py
@@ -168,7 +168,10 @@ def union_indexes(indexes, sort: bool | None = True) -> Index:
 def safe_sort_index(index: Index) -> Index:
     """
-    Returns the sorted index
+    Returns the sorted index if sorting is possible, otherwise returns the original index.
+
+    If the index contains incomparable elements (e.g., mixed types), a TypeError
+    will be caught and the original unsorted index will be returned.

     We keep the dtypes and the name attributes.
```