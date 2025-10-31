# Bug Report: pandas.core.interchange Categorical Sentinel Value Corruption

**Target**: `pandas.core.interchange.from_dataframe.categorical_column_to_series`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `categorical_column_to_series` function incorrectly handles sentinel values (specifically -1) in categorical codes by using modulo arithmetic, which silently converts null/missing values into incorrect category values instead of preserving them as NaN.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd
import numpy as np
from pandas.core.interchange.from_dataframe import from_dataframe


@settings(max_examples=500)
@given(
    n_categories=st.integers(min_value=1, max_value=10),
    codes=st.lists(
        st.one_of(
            st.integers(min_value=0, max_value=9),
            st.just(-1)
        ),
        min_size=1,
        max_size=50
    )
)
def test_categorical_roundtrip_preserves_nulls(n_categories, codes):
    categories = [f'cat{i}' for i in range(n_categories)]
    codes_arr = np.array([c if c < n_categories else -1 for c in codes], dtype=np.int8)

    cat = pd.Categorical.from_codes(codes_arr, categories=categories)
    df_original = pd.DataFrame({'col': cat})

    df_result = from_dataframe(df_original.__dataframe__())

    for i in range(len(df_original)):
        if pd.isna(df_original['col'].iloc[i]):
            assert pd.isna(df_result['col'].iloc[i]), \
                f"Expected NaN at index {i}, got '{df_result['col'].iloc[i]}'"
```

**Failing input**: Any categorical with -1 codes (sentinel values for missing data)

## Reproducing the Bug

```python
import pandas as pd
import numpy as np
from pandas.core.interchange.from_dataframe import from_dataframe

categories = pd.Index(['a', 'b', 'c'])
codes = np.array([0, 1, -1, 2], dtype=np.int8)

cat = pd.Categorical.from_codes(codes, categories=categories)
df_original = pd.DataFrame({'col': cat})

print(f"Original: {df_original['col'].values}")
print(f"Is null at index 2? {pd.isna(df_original['col'].iloc[2])}")

df_result = from_dataframe(df_original.__dataframe__())

print(f"Result: {df_result['col'].values}")
print(f"Is null at index 2? {pd.isna(df_result['col'].iloc[2])}")
```

Output:
```
Original: ['a', 'b', NaN, 'c']
Is null at index 2? True
Result: ['a', 'b', 'c', 'c']
Is null at index 2? False
```

## Why This Is A Bug

1. The original DataFrame correctly has NaN at index 2 (from the -1 sentinel code)
2. After round-trip through the interchange protocol, it becomes 'c'
3. This violates the fundamental property that data should be preserved through the interchange protocol
4. The bug silently corrupts data by replacing missing values with incorrect category values

In pandas, -1 is the standard sentinel value for missing categorical data. The modulo operation `codes % len(categories)` converts -1 to 2 (when len=3), incorrectly mapping it to a valid category instead of preserving it as null.

## Fix

```diff
--- a/pandas/core/interchange/from_dataframe.py
+++ b/pandas/core/interchange/from_dataframe.py
@@ -248,11 +248,7 @@ def categorical_column_to_series(col: Column) -> tuple[pd.Series, Any]:
         codes_buff, codes_dtype, offset=col.offset, length=col.size()
     )

-    # Doing module in order to not get ``IndexError`` for
-    # out-of-bounds sentinel values in `codes`
-    if len(categories) > 0:
-        values = categories[codes % len(categories)]
-    else:
-        values = codes
+    values = categories[codes]

     cat = pd.Categorical(
         values, categories=categories, ordered=categorical["is_ordered"]
```

The correct approach is to let pandas' `Categorical` constructor handle the codes directly. It already knows how to interpret -1 as a missing value sentinel. The modulo operation is unnecessary and incorrect - it was attempting to avoid IndexError but instead creates silent data corruption.

Alternatively, if bounds checking is desired, it should explicitly handle sentinel values:

```python
if len(categories) > 0:
    mask = codes == -1
    safe_codes = np.where(mask, 0, codes)
    values = categories[safe_codes]
    values = np.where(mask, None, values)
else:
    values = codes
```