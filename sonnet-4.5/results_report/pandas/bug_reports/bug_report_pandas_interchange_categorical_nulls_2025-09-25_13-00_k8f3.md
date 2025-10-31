# Bug Report: pandas.api.interchange Categorical Nulls Lost in Round-Trip

**Target**: `pandas.api.interchange.from_dataframe`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Null values in categorical columns are lost when round-tripping through the DataFrame interchange protocol. The `from_dataframe` function incorrectly maps null sentinel values to valid category values due to a modulo operation that wraps out-of-bounds indices.

## Property-Based Test

```python
import pandas as pd
import pandas.api.interchange as interchange
from hypothesis import given, strategies as st, settings, assume


@given(st.lists(st.sampled_from(['a', 'b', 'c']), min_size=1, max_size=50),
       st.lists(st.booleans(), min_size=1, max_size=50))
@settings(max_examples=200)
def test_from_dataframe_categorical_with_nulls(cat_values, null_mask):
    assume(len(cat_values) == len(null_mask))

    values_with_nulls = [None if null else val for val, null in zip(cat_values, null_mask)]
    df = pd.DataFrame({'cat': pd.Categorical(values_with_nulls)})

    interchange_obj = df.__dataframe__()
    result = interchange.from_dataframe(interchange_obj)

    pd.testing.assert_frame_equal(result.reset_index(drop=True), df.reset_index(drop=True), check_dtype=False, check_categorical=False)
```

**Failing input**: `cat_values=['a', 'a']`, `null_mask=[False, True]`

## Reproducing the Bug

```python
import pandas as pd
import pandas.api.interchange as interchange

df = pd.DataFrame({'cat': pd.Categorical(['a', None])})
print("Original DataFrame:")
print(df)

interchange_obj = df.__dataframe__()
result = interchange.from_dataframe(interchange_obj)

print("\nResult DataFrame after round-trip:")
print(result)

assert result.iloc[1, 0] is None or pd.isna(result.iloc[1, 0])
```

Expected output:
```
Original DataFrame:
   cat
0    a
1  NaN

Result DataFrame after round-trip:
   cat
0    a
1  NaN
```

Actual output:
```
Original DataFrame:
   cat
0    a
1  NaN

Result DataFrame after round-trip:
  cat
0   a
1   a  # BUG: Should be NaN
```

## Why This Is A Bug

This violates the fundamental requirement of the interchange protocol: data should be preserved through round-trip conversions. Null values are valid data that must be maintained. This bug causes **silent data corruption** where users lose information about missing values in categorical columns.

The bug occurs in `categorical_column_to_series` in `from_dataframe.py` at line 254:

```python
values = categories[codes % len(categories)]
```

The modulo operation wraps out-of-bounds sentinel values (e.g., -1 indicating null) to valid category indices. For example, with one category, `-1 % 1 = 0`, mapping the null to the first category. While `set_nulls` is called afterwards (line 263), it cannot properly restore nulls because the Categorical has already been created with incorrect values.

## Fix

The fix is to handle null values before indexing into categories. Replace lines 251-256 with:

```diff
diff --git a/pandas/core/interchange/from_dataframe.py b/pandas/core/interchange/from_dataframe.py
index 1234567..abcdefg 100644
--- a/pandas/core/interchange/from_dataframe.py
+++ b/pandas/core/interchange/from_dataframe.py
@@ -248,12 +248,17 @@ def categorical_column_to_series(col: Column) -> tuple[pd.Series, Any]:
         codes_buff, codes_dtype, offset=col.offset, length=col.size()
     )

-    # Doing module in order to not get ``IndexError`` for
-    # out-of-bounds sentinel values in `codes`
+    # Mask out-of-bounds codes (which represent nulls) before indexing
     if len(categories) > 0:
-        values = categories[codes % len(categories)]
+        # Create a mask for valid codes
+        valid_mask = (codes >= 0) & (codes < len(categories))
+        values = np.empty(len(codes), dtype=object)
+        values[valid_mask] = categories[codes[valid_mask]]
+        values[~valid_mask] = None
     else:
         values = codes

     cat = pd.Categorical(
         values, categories=categories, ordered=categorical["is_ordered"]
     )
```

Note: The `set_nulls` call may still be needed for bitmask/bytemask null representations, but this fix ensures sentinel-based nulls are handled correctly before creating the Categorical.