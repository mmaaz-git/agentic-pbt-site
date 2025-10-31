# Bug Report: pandas.api.interchange Categorical Null Values Lost in Round-Trip

**Target**: `pandas.core.interchange.from_dataframe.categorical_column_to_series`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Null values in categorical columns are incorrectly converted to the first category value when using the DataFrame interchange protocol round-trip conversion.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd
from pandas.api.interchange import from_dataframe


@given(
    st.lists(st.sampled_from(["cat1", "cat2", "cat3", None]), min_size=0, max_size=100),
    st.booleans()
)
def test_round_trip_categorical(cat_list, ordered):
    df = pd.DataFrame({"col": pd.Categorical(cat_list, ordered=ordered)})
    result = from_dataframe(df.__dataframe__())
    pd.testing.assert_frame_equal(result, df)
```

**Failing input**: `cat_list=['cat1', None], ordered=False`

## Reproducing the Bug

```python
import pandas as pd
from pandas.api.interchange import from_dataframe

df = pd.DataFrame({"col": pd.Categorical(["cat1", None])})
print("Original:", df["col"].tolist())

result = from_dataframe(df.__dataframe__())
print("After round-trip:", result["col"].tolist())
```

**Output:**
```
Original: ['cat1', nan]
After round-trip: ['cat1', 'cat1']
```

## Why This Is A Bug

The DataFrame interchange protocol is designed to preserve data through round-trip conversions. The function `from_dataframe` should maintain null values in categorical columns, but instead converts sentinel values (`-1`) to the first category.

The issue occurs in `categorical_column_to_series` (lines 251-256 of `from_dataframe.py`):

```python
if len(categories) > 0:
    values = categories[codes % len(categories)]
else:
    values = codes
```

When a null value is represented by code `-1`, the modulo operation `-1 % len(categories)` wraps it to `0`, mapping it to `categories[0]` instead of preserving it as a sentinel for later null handling by `set_nulls`.

## Fix

The fix is to handle sentinel values before the modulo operation:

```diff
--- a/pandas/core/interchange/from_dataframe.py
+++ b/pandas/core/interchange/from_dataframe.py
@@ -248,10 +248,20 @@ def categorical_column_to_series(col: Column) -> tuple[pd.Series, Any]:
         codes_buff, codes_dtype, offset=col.offset, length=col.size()
     )

-    # Doing module in order to not get ``IndexError`` for
-    # out-of-bounds sentinel values in `codes`
+    null_kind, sentinel_val = col.describe_null
+
+    # Create values array, handling sentinel values for nulls
     if len(categories) > 0:
-        values = categories[codes % len(categories)]
+        # Use modulo for valid codes, but preserve sentinel values
+        if null_kind == ColumnNullType.USE_SENTINEL:
+            # Create mask for valid (non-sentinel) codes
+            valid_mask = codes != sentinel_val
+            values = np.empty(len(codes), dtype=object)
+            values[valid_mask] = categories[codes[valid_mask] % len(categories)]
+            values[~valid_mask] = None
+        else:
+            # No sentinel, use modulo for all codes
+            values = categories[codes % len(categories)]
     else:
         values = codes
```