# Bug Report: pandas.core.interchange Categorical Sentinel Value Corruption

**Target**: `pandas.core.interchange.from_dataframe.categorical_column_to_series`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The categorical interchange code incorrectly uses modulo to handle out-of-bounds codes, which corrupts the sentinel value (-1) used to represent missing data, converting NaN values to valid category values.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
import pandas as pd
from pandas.api.interchange import from_dataframe

@given(
    n_categories=st.integers(min_value=1, max_value=10),
    n_missing=st.integers(min_value=1, max_value=5),
)
def test_categorical_roundtrip_preserves_missing(n_categories, n_missing):
    categories = [f"cat_{i}" for i in range(n_categories)]
    codes = [-1] * n_missing + list(range(n_categories))

    cat = pd.Categorical.from_codes(codes, categories=categories)
    df = pd.DataFrame({'col': cat})

    result = from_dataframe(df.__dataframe__(allow_copy=True), allow_copy=True)

    for i in range(len(df)):
        assert pd.isna(df['col'].iloc[i]) == pd.isna(result['col'].iloc[i])
```

**Failing input**: `n_categories=3, n_missing=1` (any values trigger the bug)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import numpy as np
import pandas as pd
from pandas.api.interchange import from_dataframe

categories = ['A', 'B', 'C']
codes = np.array([0, 1, -1], dtype='int8')
cat = pd.Categorical.from_codes(codes, categories=categories)
df = pd.DataFrame({'col': cat})

print("Original:", df['col'].tolist())
print("  Index 2 is NaN:", pd.isna(df['col'].iloc[2]))

result = from_dataframe(df.__dataframe__(allow_copy=True), allow_copy=True)

print("After interchange:", result['col'].tolist())
print("  Index 2 is NaN:", pd.isna(result['col'].iloc[2]))
```

**Expected output:**
```
Original: ['A', 'B', nan]
  Index 2 is NaN: True
After interchange: ['A', 'B', nan]
  Index 2 is NaN: True
```

**Actual output:**
```
Original: ['A', 'B', nan]
  Index 2 is NaN: True
After interchange: ['A', 'B', 'C']
  Index 2 is NaN: False
```

## Why This Is A Bug

1. The sentinel value for missing categorical data is explicitly defined as -1 in `column.py:60`
2. The code at `from_dataframe.py:253-254` uses modulo: `values = categories[codes % len(categories)]`
3. With 3 categories, `-1 % 3 = 2` in Python, so the sentinel maps to `categories[2]` instead of remaining as -1
4. The subsequent `set_nulls()` call at line 263 checks if `data == sentinel_val`, but data now contains the category string (e.g., 'C'), not -1
5. This causes missing values to be incorrectly converted to valid categories

This is data corruption that silently converts missing values to actual data, which is a serious integrity issue.

## Fix

The fix is to check for and preserve sentinel values before applying the modulo operation:

```diff
--- a/pandas/core/interchange/from_dataframe.py
+++ b/pandas/core/interchange/from_dataframe.py
@@ -248,10 +248,17 @@ def categorical_column_to_series(col: Column) -> tuple[pd.Series, Any]:
         codes_buff, codes_dtype, offset=col.offset, length=col.size()
     )

-    # Doing module in order to not get ``IndexError`` for
-    # out-of-bounds sentinel values in `codes`
+    # Handle sentinel values for missing data before mapping codes to categories
+    null_kind, sentinel_val = col.describe_null
+    mask = None
+    if null_kind == ColumnNullType.USE_SENTINEL and sentinel_val is not None:
+        mask = codes == sentinel_val
+        # Use 0 as temporary placeholder for sentinel positions to avoid IndexError
+        codes = np.where(mask, 0, codes)
+
+    # Map codes to categories, using modulo for any remaining out-of-bounds codes
     if len(categories) > 0:
         values = categories[codes % len(categories)]
+        # Restore sentinel positions after mapping
+        if mask is not None:
+            values = np.where(mask, None, values)
     else:
         values = codes
@@ -261,7 +268,6 @@ def categorical_column_to_series(col: Column) -> tuple[pd.Series, Any]:
     )
     data = pd.Series(cat)

-    data = set_nulls(data, col, buffers["validity"])
     return data, buffers
```

Alternatively, a simpler fix would be to handle sentinel values after categorical creation but before set_nulls, or to not use modulo at all and let invalid codes raise an error (which would be more correct behavior).