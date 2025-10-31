# Bug Report: pandas.core.interchange Categorical Null Values Lost

**Target**: `pandas.core.interchange.from_dataframe.categorical_column_to_series`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When converting categorical data with null values through the DataFrame interchange protocol, null values are incorrectly mapped to actual category values instead of being preserved as NaN. This causes silent data corruption where null values become non-null categorical values.

## Property-Based Test

```python
import pandas as pd
from pandas.api.interchange import from_dataframe
from hypothesis import given, strategies as st, settings

@given(st.data())
@settings(max_examples=100)
def test_categorical_with_nulls_property(data):
    categories = data.draw(st.lists(
        st.text(min_size=1, max_size=5, alphabet=st.characters(min_codepoint=ord('a'), max_codepoint=ord('z'))),
        min_size=1,
        max_size=5,
        unique=True
    ))
    n_rows = data.draw(st.integers(min_value=1, max_value=20))

    values = data.draw(st.lists(
        st.one_of(
            st.sampled_from(categories),
            st.none()
        ),
        min_size=n_rows,
        max_size=n_rows
    ))

    df = pd.DataFrame({"cat": pd.Categorical(values, categories=categories)})

    null_count_before = df["cat"].isna().sum()

    df_interchange = df.__dataframe__()
    df_result = from_dataframe(df_interchange)

    null_count_after = df_result["cat"].isna().sum()

    assert null_count_before == null_count_after
```

**Failing input**: `categories=['a'], values=[None]`

## Reproducing the Bug

```python
import pandas as pd
from pandas.api.interchange import from_dataframe

df = pd.DataFrame({
    "cat": pd.Categorical(["a", "b", None, "c"], categories=["a", "b", "c"])
})

print("Original:")
print(df)
print(f"Null values: {df['cat'].isna().tolist()}")

df_result = from_dataframe(df.__dataframe__())

print("\nAfter interchange:")
print(df_result)
print(f"Null values: {df_result['cat'].isna().tolist()}")
```

**Output:**
```
Original:
   cat
0    a
1    b
2  NaN
3    c
Null values: [False, False, True, False]

After interchange:
  cat
0   a
1   b
2   c
3   c
Null values: [False, False, False, False]
```

## Why This Is A Bug

Pandas categoricals use -1 as a sentinel value to represent null/missing values. In the interchange protocol conversion, the code at `from_dataframe.py:254` uses modulo arithmetic to avoid IndexError:

```python
if len(categories) > 0:
    values = categories[codes % len(categories)]
```

This causes -1 (null sentinel) to be mapped to `categories[-1 % len(categories)]`, which equals `categories[len(categories) - 1]` (the last category). For example, with 3 categories:
- Code -1 → -1 % 3 = 2 → categories[2] (last category)
- Code 5 → 5 % 3 = 2 → categories[2]

This silently converts null values into actual category values, causing data corruption. Even though `set_nulls` is called later (line 263), it cannot recover the original null values because they've already been replaced with actual category values.

## Fix

The modulo operation should be removed. Instead, handle sentinel values explicitly:

```diff
diff --git a/pandas/core/interchange/from_dataframe.py b/pandas/core/interchange/from_dataframe.py
index 1234567..abcdefg 100644
--- a/pandas/core/interchange/from_dataframe.py
+++ b/pandas/core/interchange/from_dataframe.py
@@ -248,11 +248,18 @@ def categorical_column_to_series(col: Column) -> tuple[pd.Series, Any]:
         codes_buff, codes_dtype, offset=col.offset, length=col.size()
     )

-    # Doing module in order to not get ``IndexError`` for
-    # out-of-bounds sentinel values in `codes`
+    # Handle sentinel values by temporarily replacing them to avoid IndexError.
+    # The actual null handling is done by set_nulls() later.
+    null_kind, sentinel_val = col.describe_null
     if len(categories) > 0:
-        values = categories[codes % len(categories)]
+        if null_kind == ColumnNullType.USE_SENTINEL:
+            # Replace sentinel values with 0 temporarily (will be set to NaN later)
+            codes_clean = np.where(codes == sentinel_val, 0, codes)
+            values = categories[codes_clean]
+        else:
+            values = categories[codes]
     else:
         values = codes