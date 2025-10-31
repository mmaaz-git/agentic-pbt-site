# Bug Report: pandas.api.interchange Categorical Null Value Loss

**Target**: `pandas.api.interchange.from_dataframe`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When converting a pandas DataFrame with categorical columns containing null values through the interchange protocol, null values are incorrectly mapped to valid category values instead of being preserved as NaN.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st, settings


@given(st.lists(st.integers(min_value=-1, max_value=2), min_size=5, max_size=20))
@settings(max_examples=100)
def test_categorical_null_preservation(codes):
    categories = ['a', 'b', 'c']
    df = pd.DataFrame({'cat': pd.Categorical.from_codes(codes, categories=categories)})

    interchange_obj = df.__dataframe__()
    result = pd.api.interchange.from_dataframe(interchange_obj)

    assert result['cat'].isna().sum() == df['cat'].isna().sum(), \
        f"Null count mismatch: {result['cat'].isna().sum()} != {df['cat'].isna().sum()}"
```

**Failing input**: `codes=[0, 0, 0, 0, -1]`

## Reproducing the Bug

```python
import pandas as pd

categories = ['a', 'b', 'c']
codes = [0, -1, 2]
df = pd.DataFrame({'cat': pd.Categorical.from_codes(codes, categories=categories)})

print("Original DataFrame:")
print(df)
print(f"Null values: {df['cat'].isna().sum()}")

interchange_obj = df.__dataframe__()
result = pd.api.interchange.from_dataframe(interchange_obj)

print("\nResult after round-trip:")
print(result)
print(f"Null values: {result['cat'].isna().sum()}")

assert list(df['cat']) == list(result['cat'])
```

Expected output: `['a', NaN, 'c']` with 1 null value
Actual output: `['a', 'c', 'c']` with 0 null values

## Why This Is A Bug

In pandas categorical encoding, the code `-1` represents a null value. When a DataFrame with categorical data is converted through the interchange protocol and back, null values should be preserved. Instead, the modulo operation in `categorical_column_to_series` at lines 253-256 of `from_dataframe.py` wraps the -1 code around:

```python
if len(categories) > 0:
    values = categories[codes % len(categories)]
```

Since `-1 % 3 = 2`, the null value is incorrectly mapped to `categories[2] = 'c'`. This silently corrupts data by converting null values to valid category values, which is a serious logic error that can lead to incorrect analysis results.

## Fix

```diff
--- a/pandas/core/interchange/from_dataframe.py
+++ b/pandas/core/interchange/from_dataframe.py
@@ -247,12 +247,17 @@ def categorical_column_to_series(col: Column) -> tuple[pd.Series, Any]:
         codes_buff, codes_dtype, offset=col.offset, length=col.size()
     )

-    # Doing module in order to not get ``IndexError`` for
-    # out-of-bounds sentinel values in `codes`
+    # Handle null values (-1 codes) and validate non-null codes
     if len(categories) > 0:
-        values = categories[codes % len(categories)]
+        # Create a mask for null values (code == -1)
+        null_mask = codes == -1
+        # For valid codes, map to categories; for null codes, use placeholder
+        valid_codes = np.where(null_mask, 0, codes)
+        values = categories[valid_codes]
+        # Mark null positions with NaN
+        values = values.astype(object)
+        values[null_mask] = np.nan
     else:
-        values = codes
+        values = np.full(len(codes), np.nan, dtype=object)

     cat = pd.Categorical(
         values, categories=categories, ordered=categorical["is_ordered"]
```

Note: This fix handles the -1 sentinel value case. A more robust solution would also validate that all non-null codes are within the valid range [0, len(categories)-1] and raise an appropriate error for truly out-of-bounds values, rather than silently wrapping them with modulo.