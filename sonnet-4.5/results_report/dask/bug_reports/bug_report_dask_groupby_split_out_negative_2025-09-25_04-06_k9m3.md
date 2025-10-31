# Bug Report: dask.dataframe.dask_expr Groupby Split Calculation Returns Negative Value

**Target**: `dask.dataframe.dask_expr._groupby._adjust_split_out_for_group_keys`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The function `_adjust_split_out_for_group_keys` can return negative values when called with an empty `by` list (`len(by) == 0`), due to missing input validation. This function is used to calculate the number of output partitions for groupby operations.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import math

def _adjust_split_out_for_group_keys(npartitions, by):
    if len(by) == 1:
        return math.ceil(npartitions / 15)
    return math.ceil(npartitions / (10 / (len(by) - 1)))

@given(
    st.integers(min_value=1, max_value=1000),
    st.lists(st.text(), max_size=10)
)
def test_split_out_is_positive(npartitions, by):
    result = _adjust_split_out_for_group_keys(npartitions, by)
    assert result > 0, f"Expected positive split_out, got {result}"
```

**Failing input**: `npartitions=100, by=[]`

## Reproducing the Bug

```python
import math

def _adjust_split_out_for_group_keys(npartitions, by):
    if len(by) == 1:
        return math.ceil(npartitions / 15)
    return math.ceil(npartitions / (10 / (len(by) - 1)))

npartitions = 100
by = []

result = _adjust_split_out_for_group_keys(npartitions, by)
print(f"Result: {result}")

denominator = 10 / (len(by) - 1)
print(f"Denominator: {denominator}")
print(f"Expected: math.ceil({npartitions} / {denominator}) = {math.ceil(npartitions / denominator)}")
```

**Output:**
```
Result: -10
Denominator: -10.0
Expected: math.ceil(100 / -10.0) = -10
```

## Why This Is A Bug

The function `_adjust_split_out_for_group_keys` is designed to calculate an appropriate number of output partitions for groupby operations based on the number of grouping keys. When `len(by) == 0`, the formula `npartitions / (10 / (len(by) - 1))` becomes `npartitions / (10 / -1)` = `npartitions / -10`, resulting in a negative value.

While groupby operations with zero keys may not be a valid use case, the function should either:
1. Validate inputs and raise an appropriate error
2. Handle the edge case gracefully with a reasonable default

Returning a negative number of partitions violates the function's contract and could lead to confusing errors downstream.

## Fix

Add input validation to handle the edge case:

```diff
--- a/dask/dataframe/dask_expr/_groupby.py
+++ b/dask/dataframe/dask_expr/_groupby.py
@@ -94,6 +94,8 @@ def _as_dict(key, value):


 def _adjust_split_out_for_group_keys(npartitions, by):
+    if len(by) == 0:
+        raise ValueError("Cannot adjust split_out for empty 'by' list")
     if len(by) == 1:
         return math.ceil(npartitions / 15)
     return math.ceil(npartitions / (10 / (len(by) - 1)))
```

Alternatively, return a sensible default:

```diff
--- a/dask/dataframe/dask_expr/_groupby.py
+++ b/dask/dataframe/dask_expr/_groupby.py
@@ -94,6 +94,8 @@ def _as_dict(key, value):


 def _adjust_split_out_for_group_keys(npartitions, by):
+    if len(by) == 0:
+        return 1
     if len(by) == 1:
         return math.ceil(npartitions / 15)
     return math.ceil(npartitions / (10 / (len(by) - 1)))
```