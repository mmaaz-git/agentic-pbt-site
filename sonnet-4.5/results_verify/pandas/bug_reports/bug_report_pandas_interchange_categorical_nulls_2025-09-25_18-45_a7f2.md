# Bug Report: pandas.core.interchange Categorical Null Values Corrupted

**Target**: `pandas.core.interchange.from_dataframe.categorical_column_to_series`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `categorical_column_to_series` function incorrectly handles null values in categorical columns. Null values (represented by code -1) are converted to actual category values instead of being preserved as nulls, causing silent data corruption.

## Property-Based Test

```python
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from pandas.core.interchange.from_dataframe import from_dataframe


@given(
    categories=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=10, unique=True),
    values_with_nulls=st.lists(
        st.one_of(st.integers(min_value=0, max_value=9), st.none()),
        min_size=1,
        max_size=20
    )
)
@settings(max_examples=300)
def test_categorical_nulls_preserved(categories, values_with_nulls):
    assume(len(categories) > 0)

    cat_values = []
    for v in values_with_nulls:
        if v is None:
            cat_values.append(None)
        else:
            cat_values.append(categories[v % len(categories)])

    try:
        cat_data = pd.Categorical(cat_values, categories=categories)
        df = pd.DataFrame({'cat': cat_data})

        original_null_count = df['cat'].isna().sum()

        result = from_dataframe(df.__dataframe__())

        result_null_count = result['cat'].isna().sum()

        if original_null_count != result_null_count:
            raise AssertionError(
                f"Null count changed! Original: {original_null_count}, "
                f"Result: {result_null_count}"
            )
    except Exception as e:
        if "Null count changed" in str(e):
            raise
        pass
```

**Failing input**: `categories=['0'], values_with_nulls=[None]`

## Reproducing the Bug

```python
import pandas as pd
from pandas.core.interchange.from_dataframe import from_dataframe

cat_data = pd.Categorical(['a', 'b', None, 'a'], categories=['a', 'b', 'c'])
df = pd.DataFrame({'cat': cat_data})

print("Original DataFrame:")
print(df)
print(f"Original categorical codes: {df['cat'].cat.codes.tolist()}")

result = from_dataframe(df.__dataframe__())

print("\nResult DataFrame:")
print(result)
print(f"Result categorical codes: {result['cat'].cat.codes.tolist()}")

print(f"\nOriginal nulls: {df['cat'].isna().tolist()}")
print(f"Result nulls: {result['cat'].isna().tolist()}")
```

Output:
```
Original DataFrame:
   cat
0    a
1    b
2  NaN
3    a
Original categorical codes: [0, 1, -1, 0]

Result DataFrame:
  cat
0   a
1   b
2   c
3   a
Result categorical codes: [0, 1, 2, 0]

Original nulls: [False, False, True, False]
Result nulls: [False, False, False, False]
```

The null value (code -1) at index 2 is incorrectly converted to category 'c' (code 2).

## Why This Is A Bug

Categorical columns in pandas use -1 as a sentinel value to represent null/missing values. The current implementation uses modulo arithmetic to prevent IndexError on out-of-bounds codes:

```python
if len(categories) > 0:
    values = categories[codes % len(categories)]
```

However, `-1 % len(categories)` does not equal -1 in Python; it wraps around to `len(categories) - 1`. For example:
- If categories has 3 elements: `-1 % 3 = 2`
- This maps null values to `categories[2]`, which is a real category value

This causes silent data corruption where null values are replaced with actual category values.

## Fix

The fix should handle sentinel values (-1) before the modulo operation:

```diff
--- a/pandas/core/interchange/from_dataframe.py
+++ b/pandas/core/interchange/from_dataframe.py
@@ -248,10 +248,14 @@ def categorical_column_to_series(col: Column) -> tuple[pd.Series, Any]:
         codes_buff, codes_dtype, offset=col.offset, length=col.size()
     )

-    # Doing module in order to not get ``IndexError`` for
-    # out-of-bounds sentinel values in `codes`
+    # Handle null sentinel values (-1) separately to prevent them
+    # from being mapped to actual categories via modulo arithmetic
     if len(categories) > 0:
-        values = categories[codes % len(categories)]
+        # Create a mask for valid (non-null) codes
+        valid_mask = codes >= 0
+        values = np.empty(len(codes), dtype=object)
+        values[valid_mask] = categories[codes[valid_mask] % len(categories)]
+        values[~valid_mask] = None
     else:
         values = codes
```

This fix explicitly checks for sentinel values (negative codes) and handles them as null, while still using modulo for valid codes to prevent IndexError on out-of-bounds positive values.