# Bug Report: pandas.core.interchange Categorical Null Values Corrupted

**Target**: `pandas.core.interchange.from_dataframe.categorical_column_to_series`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Converting categorical data with null values through the DataFrame interchange protocol corrupts null values, replacing them with valid category values. The modulo operation used to handle out-of-bounds codes inadvertently transforms the null sentinel value (-1) into a valid category index.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd
from pandas.core.interchange.from_dataframe import from_dataframe

@given(
    st.lists(st.sampled_from(['cat1', 'cat2', 'cat3', None]), min_size=1, max_size=50)
)
def test_round_trip_categorical(data):
    df = pd.DataFrame({'A': pd.Categorical(data)})
    df_xchg = df.__dataframe__(allow_copy=True)
    result = from_dataframe(df_xchg, allow_copy=True)
    pd.testing.assert_frame_equal(df, result)
```

**Failing input**: `data=['cat1', None]`

## Reproducing the Bug

```python
import pandas as pd
from pandas.core.interchange.from_dataframe import from_dataframe

data = ['cat1', None]
df = pd.DataFrame({'A': pd.Categorical(data)})

print("Original:")
print(f"  Values: {df['A'].tolist()}")
print(f"  Codes: {df['A'].cat.codes.tolist()}")

df_xchg = df.__dataframe__(allow_copy=True)
result = from_dataframe(df_xchg, allow_copy=True)

print("\nAfter round-trip:")
print(f"  Values: {result['A'].tolist()}")
print(f"  Codes: {result['A'].cat.codes.tolist()}")

print("\nExpected: ['cat1', nan]")
print(f"Actual:   {result['A'].tolist()}")
```

Output:
```
Original:
  Values: ['cat1', nan]
  Codes: [0, -1]

After round-trip:
  Values: ['cat1', 'cat1']
  Codes: [0, 0]

Expected: ['cat1', nan]
Actual:   ['cat1', 'cat1']
```

## Why This Is A Bug

The interchange protocol specifies that categorical columns use a sentinel value of `-1` to represent null values (see `ColumnNullType.USE_SENTINEL`). However, in `categorical_column_to_series` (from_dataframe.py:251-256), the modulo operation `codes % len(categories)` transforms this sentinel value into a valid category index:

- When `codes = [0, -1]` and `len(categories) = 1`
- `codes % 1` produces `[0, 0]` (since `-1 % 1 = 0`)
- The null value is now indistinguishable from the first category

The `set_nulls` function cannot fix this because:
1. Categorical columns use `USE_SENTINEL` null representation (no separate validity buffer)
2. The sentinel value is in the data buffer itself
3. The modulo operation has already corrupted the sentinel, making it look like valid data

## Fix

```diff
diff --git a/pandas/core/interchange/from_dataframe.py b/pandas/core/interchange/from_dataframe.py
index 1234567..abcdefg 100644
--- a/pandas/core/interchange/from_dataframe.py
+++ b/pandas/core/interchange/from_dataframe.py
@@ -248,12 +248,18 @@ def categorical_column_to_series(col: Column) -> tuple[pd.Series, Any]:
         codes_buff, codes_dtype, offset=col.offset, length=col.size()
     )

-    # Doing module in order to not get ``IndexError`` for
-    # out-of-bounds sentinel values in `codes`
+    # Handle null sentinel values before indexing into categories
+    null_kind, sentinel_val = col.describe_null
+    if null_kind == ColumnNullType.USE_SENTINEL:
+        null_mask = codes == sentinel_val
+    else:
+        null_mask = None
+
+    # Use modulo to handle out-of-bounds values (but not sentinel values)
     if len(categories) > 0:
-        values = categories[codes % len(categories)]
+        # Set sentinel values to 0 temporarily to avoid modulo issues
+        safe_codes = np.where(null_mask, 0, codes) if null_mask is not None else codes
+        values = categories[safe_codes % len(categories)]
     else:
         values = codes

@@ -261,7 +267,11 @@ def categorical_column_to_series(col: Column) -> tuple[pd.Series, Any]:
         values, categories=categories, ordered=categorical["is_ordered"]
     )
     data = pd.Series(cat)
-
+
+    # Restore nulls for USE_SENTINEL case
+    if null_mask is not None and np.any(null_mask):
+        data = data.copy()
+        data[null_mask] = None
+
     data = set_nulls(data, col, buffers["validity"])
     return data, buffers
```