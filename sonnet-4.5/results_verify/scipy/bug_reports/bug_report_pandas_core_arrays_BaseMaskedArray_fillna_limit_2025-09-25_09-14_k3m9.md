# Bug Report: BaseMaskedArray.fillna Ignores limit Parameter

**Target**: `pandas.core.arrays.BaseMaskedArray.fillna`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `fillna` method in `BaseMaskedArray` (affecting `IntegerArray`, `FloatingArray`, `BooleanArray`) completely ignores the `limit` parameter when filling with a scalar value, filling all NA values instead of respecting the specified limit.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from pandas.core.arrays import IntegerArray

@st.composite
def integer_arrays(draw):
    data = draw(st.lists(st.one_of(st.integers(-100, 100), st.none()), min_size=2, max_size=50))
    return IntegerArray._from_sequence(data, dtype="Int64")

@given(integer_arrays(), st.integers(0, 5))
def test_fillna_limit_bounds_integer(arr, limit):
    assume(arr.isna().any())

    result = arr.fillna(value=0, limit=limit)
    na_filled = arr.isna().sum() - result.isna().sum()

    assert na_filled <= limit
```

**Failing input**: `arr=[None, None, None]`, `limit=1`

## Reproducing the Bug

```python
from pandas.core.arrays import IntegerArray

arr = IntegerArray._from_sequence([None, None, None], dtype="Int64")
print(f"Original: {arr}, NA count: {arr.isna().sum()}")

result = arr.fillna(value=999, limit=1)
print(f"After fillna(value=999, limit=1): {result}")
print(f"NA count: {result.isna().sum()}")

assert result.isna().sum() == 2, "Expected limit=1 to fill only 1 NA"
```

Output:
```
Original: [<NA>, <NA>, <NA>], NA count: 3
After fillna(value=999, limit=1): [999, 999, 999]
NA count: 0
AssertionError: Expected limit=1 to fill only 1 NA
```

## Why This Is A Bug

According to the `fillna` docstring:

> If method is not specified, this is the maximum number of entries along the entire axis where NaNs will be filled.

When `limit=1` is specified with a value, only 1 NA should be filled. When `limit=0`, zero NAs should be filled. However, the implementation ignores the `limit` parameter entirely when `method=None`, filling all NA values.

This is also inconsistent: using `method='ffill'` with `limit=0` raises `ValueError: Limit must be greater than 0`, but using `value` with `limit=0` is accepted and fills all NAs.

The bug affects user-facing pandas Series API and silently violates user expectations.

## Fix

```diff
--- a/pandas/core/arrays/masked.py
+++ b/pandas/core/arrays/masked.py
@@ -246,6 +246,17 @@ class BaseMaskedArray(ExtensionArray):
             else:
                 # fill with value
+                if limit is not None:
+                    # Apply limit when filling with a value
+                    indices = np.where(mask)[0]
+                    if limit < len(indices):
+                        indices = indices[:limit]
+                    fill_mask = np.zeros_like(mask)
+                    fill_mask[indices] = True
+                else:
+                    fill_mask = mask
+
                 if copy:
                     new_values = self.copy()
                 else:
                     new_values = self[:]
-                new_values[mask] = value
+                new_values[fill_mask] = value
         else:
             if copy:
                 new_values = self.copy()
```