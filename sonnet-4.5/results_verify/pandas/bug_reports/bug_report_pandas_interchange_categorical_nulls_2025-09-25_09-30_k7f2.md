# Bug Report: pandas.api.interchange Categorical Null Values Lost

**Target**: `pandas.api.interchange.from_dataframe`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Null values in categorical columns are incorrectly converted to actual category values when round-tripping through the DataFrame interchange protocol. This causes silent data corruption.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd
from pandas.api.interchange import from_dataframe


@given(
    st.lists(st.sampled_from(["a", "b", "c", None]), min_size=1, max_size=50)
)
def test_categorical_with_nulls(cat_values):
    df = pd.DataFrame({"cat_col": pd.Categorical(cat_values)})

    interchange_df = df.__dataframe__()
    result_df = from_dataframe(interchange_df)

    pd.testing.assert_series_equal(
        result_df["cat_col"].reset_index(drop=True),
        df["cat_col"].reset_index(drop=True)
    )
```

**Failing input**: `cat_values=['a', None]`

## Reproducing the Bug

```python
import pandas as pd
from pandas.api.interchange import from_dataframe

df = pd.DataFrame({"cat_col": pd.Categorical(['a', None])})
print("Original:", list(df['cat_col']))

interchange_df = df.__dataframe__()
result_df = from_dataframe(interchange_df)

print("After round-trip:", list(result_df['cat_col']))
```

Output:
```
Original: ['a', NaN]
After round-trip: ['a', 'a']
```

## Why This Is A Bug

The interchange protocol specifies that categorical null values are represented as `-1` sentinel values in the codes array (see `pandas/core/interchange/column.py:60`). However, in `categorical_column_to_series` (from_dataframe.py:254), the code uses modulo arithmetic to prevent IndexError:

```python
if len(categories) > 0:
    values = categories[codes % len(categories)]
```

This causes `-1 % len(categories)` to wrap around to a valid index. For example, with a single category `['a']`:
- `-1 % 1 = 0`, which maps to category `'a'`

This silently converts null values to the first category, causing data corruption.

## Fix

The fix should check for sentinel values before applying the modulo operation:

```diff
--- a/pandas/core/interchange/from_dataframe.py
+++ b/pandas/core/interchange/from_dataframe.py
@@ -251,8 +251,13 @@ def categorical_column_to_series(col: Column) -> tuple[pd.Series, Any]:
     # Doing module in order to not get ``IndexError`` for
     # out-of-bounds sentinel values in `codes`
     if len(categories) > 0:
-        values = categories[codes % len(categories)]
+        # Preserve null sentinel values (-1) before modulo operation
+        null_mask = codes == -1
+        safe_codes = codes.copy()
+        safe_codes[null_mask] = 0  # Use any valid index temporarily
+        values = categories[safe_codes % len(categories)]
+        values[null_mask] = None  # Restore nulls
     else:
         values = codes

     cat = pd.Categorical(