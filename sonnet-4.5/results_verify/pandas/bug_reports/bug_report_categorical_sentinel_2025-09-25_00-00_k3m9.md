# Bug Report: pandas.core.interchange Categorical Missing Value Corruption

**Target**: `pandas.core.interchange.from_dataframe.categorical_column_to_series`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The interchange protocol corrupts categorical data with missing values. Code -1 (the pandas sentinel for missing categorical values) is incorrectly mapped to a valid category instead of being preserved as NaN.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd

@given(st.lists(st.text(min_size=1, max_size=5), min_size=1, max_size=10))
def test_categorical_preserves_missing_values(categories):
    """Categorical -1 codes (missing values) should round-trip as NaN."""
    cat_data = pd.Categorical.from_codes([-1], categories=list(set(categories)), ordered=False)
    series = pd.Series(cat_data)

    from pandas.core.interchange.from_dataframe import from_dataframe
    df = pd.DataFrame({"cat": series})
    result = from_dataframe(df.__dataframe__(allow_copy=True))

    assert pd.isna(result["cat"].iloc[0]), "Missing categorical value not preserved"
```

**Failing input**: `categories=['a']`, `codes=[-1]`

## Reproducing the Bug

```python
import pandas as pd
from pandas.core.interchange.from_dataframe import categorical_column_to_series

cat_data = pd.Categorical.from_codes([-1], categories=['a'], ordered=False)
series = pd.Series(cat_data, name="cat_col")

print(f"Original: {series.iloc[0]}")
print(f"Is NaN: {pd.isna(series.iloc[0])}")

interchange_obj = pd.DataFrame({"cat_col": series}).__dataframe__(allow_copy=True)
col = interchange_obj.get_column_by_name("cat_col")
result_series, _ = categorical_column_to_series(col)

print(f"After interchange: {result_series.iloc[0]}")
print(f"Is NaN: {pd.isna(result_series.iloc[0])}")
```

**Output:**
```
Original: nan
Is NaN: True
After interchange: a
Is NaN: False
```

## Why This Is A Bug

In pandas, categorical code -1 is the sentinel value for missing data. The interchange protocol should preserve this semantic meaning. Instead, the current implementation uses modulo arithmetic (`codes % len(categories)`) which maps -1 to 0 when there's only one category:

```python
# from_dataframe.py, lines 253-256
if len(categories) > 0:
    values = categories[codes % len(categories)]  # -1 % 1 = 0!
else:
    values = codes
```

This silently corrupts data: missing values become valid category values. The comment claims this prevents IndexError for "out-of-bounds sentinel values", but it actually creates a data corruption bug.

## Fix

The issue is that the modulo operation converts -1 codes to valid categories before `set_nulls` can detect them. The fix is to preserve -1 codes as -1 values so that `set_nulls` (which checks `data == sentinel_val`) can properly identify and convert them to NaN:

```diff
--- a/pandas/core/interchange/from_dataframe.py
+++ b/pandas/core/interchange/from_dataframe.py
@@ -250,10 +250,12 @@ def categorical_column_to_series(col: Column) -> tuple[pd.Series, Any]:
         length=col.size()
     )

-    # Doing module in order to not get ``IndexError`` for
-    # out-of-bounds sentinel values in `codes`
+    # Map codes to categories, preserving -1 (missing value sentinel)
+    # Note: -1 codes will be handled by set_nulls below
     if len(categories) > 0:
-        values = categories[codes % len(categories)]
+        valid_mask = codes >= 0
+        values = np.empty(len(codes), dtype=object)
+        values[valid_mask] = categories[codes[valid_mask]]
+        values[~valid_mask] = -1
     else:
         values = codes
```

This preserves -1 sentinel values in the data, allowing `set_nulls` (line 263) to correctly identify and convert them to NaN.