# Bug Report: pandas.api.interchange Categorical Null Values Incorrectly Mapped

**Target**: `pandas.core.interchange.from_dataframe.categorical_column_to_series`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The interchange protocol's categorical column handler incorrectly maps null sentinel values (-1) to actual category values through a modulo operation, causing silent data corruption where null values become non-null categorical values.

## Property-Based Test

```python
import numpy as np
import pandas as pd
from hypothesis import given, settings, strategies as st


@given(st.data())
@settings(max_examples=100)
def test_categorical_negative_sentinel_preserved(data):
    n_categories = data.draw(st.integers(min_value=2, max_value=10))
    categories = [f"cat_{i}" for i in range(n_categories)]
    n_rows = data.draw(st.integers(min_value=1, max_value=20))

    codes = []
    for _ in range(n_rows):
        is_null = data.draw(st.booleans())
        if is_null:
            codes.append(-1)
        else:
            codes.append(data.draw(st.integers(min_value=0, max_value=n_categories-1)))

    codes = np.array(codes, dtype=np.int64)
    cat_values = pd.Categorical.from_codes(codes, categories=categories)
    df = pd.DataFrame({"cat_col": cat_values})

    xchg = df.__dataframe__()
    result = pd.api.interchange.from_dataframe(xchg)

    np.testing.assert_array_equal(
        df["cat_col"].isna().values,
        result["cat_col"].isna().values,
        err_msg="Null positions don't match after interchange"
    )
```

**Failing input**: Any categorical with sentinel value -1 for nulls

## Reproducing the Bug

```python
import numpy as np
import pandas as pd

categories = ["A", "B", "C"]
codes = np.array([-1, 0, 1, -1, 2], dtype=np.int64)

cat = pd.Categorical.from_codes(codes, categories=categories)
df = pd.DataFrame({"cat_col": cat})

print(f"Original null positions: {np.where(df['cat_col'].isna())[0]}")

xchg = df.__dataframe__()
result = pd.api.interchange.from_dataframe(xchg)

print(f"Result null positions: {np.where(result['cat_col'].isna())[0]}")

assert np.array_equal(df['cat_col'].isna().values, result['cat_col'].isna().values), \
    "Null positions changed during interchange!"
```

Output:
```
Original null positions: [0 3]
Result null positions: []
AssertionError: Null positions changed during interchange!
```

## Why This Is A Bug

This violates pandas' interchange protocol specification and causes silent data corruption:

1. **Protocol Violation**: The interchange protocol explicitly defines that categorical columns use `ColumnNullType.USE_SENTINEL` with sentinel value -1 for null values (see `pandas/core/interchange/column.py:58-60`).

2. **Silent Data Corruption**: The modulo operation on line 254 of `from_dataframe.py` wraps negative sentinel values into the valid category range:
   - Sentinel value -1 becomes index 2 (for 3 categories): `-1 % 3 = 2`
   - This maps the null to the **last category** instead of preserving it as null
   - Users receive no warning about this data corruption

3. **set_nulls Cannot Recover**: After the modulo operation, `set_nulls()` cannot identify the original sentinel values because:
   - The codes have been converted to actual category values
   - Comparing categorical values to the integer sentinel (-1) never matches
   - The null information is permanently lost

4. **Real-World Impact**: This affects any categorical data with missing values being transferred via the interchange protocol:
   - Data analysis results become incorrect
   - Statistical summaries miscount null values
   - Filtering and grouping operations produce wrong results

## Root Cause

In `pandas/core/interchange/from_dataframe.py`, lines 251-256:

```python
# Doing module in order to not get ``IndexError`` for
# out-of-bounds sentinel values in `codes`
if len(categories) > 0:
    values = categories[codes % len(categories)]
else:
    values = codes
```

The comment acknowledges that sentinel values exist in the codes, but the modulo "fix" creates a worse bug by mapping sentinels to actual categories.

## Fix

The sentinel values should be handled **before** indexing into categories, not wrapped via modulo:

```diff
--- a/pandas/core/interchange/from_dataframe.py
+++ b/pandas/core/interchange/from_dataframe.py
@@ -248,13 +248,20 @@ def categorical_column_to_series(col: Column) -> tuple[pd.Series, Any]:
         codes_buff, codes_dtype, offset=col.offset, length=col.size()
     )

-    # Doing module in order to not get ``IndexError`` for
-    # out-of-bounds sentinel values in `codes`
+    # Handle sentinel values before indexing into categories
+    # Sentinel value -1 indicates null/missing values in categorical codes
+    null_kind, sentinel_val = col.describe_null
+
     if len(categories) > 0:
-        values = categories[codes % len(categories)]
+        # Create values array, using None for sentinel values
+        values = np.empty(len(codes), dtype=object)
+        if null_kind == ColumnNullType.USE_SENTINEL:
+            valid_mask = codes != sentinel_val
+            values[valid_mask] = categories[codes[valid_mask]]
+            values[~valid_mask] = None
+        else:
+            values[:] = categories[codes]
     else:
         values = codes

     cat = pd.Categorical(
         values, categories=categories, ordered=categorical["is_ordered"]
     )
```

This fix:
1. Checks for sentinel values before indexing
2. Only indexes valid codes into categories
3. Sets sentinel positions to None explicitly
4. Preserves null information for `set_nulls` to work correctly