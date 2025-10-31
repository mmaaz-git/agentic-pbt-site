# Bug Report: pandas.api.interchange Categorical Null Values Incorrectly Converted

**Target**: `pandas.api.interchange.from_dataframe`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When converting categorical data through the DataFrame interchange protocol, null values (represented by code -1) are incorrectly mapped to actual category values instead of being preserved as nulls. This is caused by a modulo operation that wraps -1 to 0, silently corrupting data.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st, settings
from pandas.api.interchange import from_dataframe

@given(st.data())
@settings(max_examples=200)
def test_categorical_with_nulls(data):
    nrows = data.draw(st.integers(min_value=1, max_value=20))
    categories = data.draw(st.lists(
        st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))),
        min_size=1, max_size=5, unique=True
    ))

    codes = data.draw(st.lists(
        st.one_of(
            st.just(-1),
            st.integers(min_value=0, max_value=len(categories) - 1)
        ),
        min_size=nrows, max_size=nrows
    ))

    cat = pd.Categorical.from_codes(codes, categories=categories)
    df = pd.DataFrame({'col': cat})

    interchange_obj = df.__dataframe__()
    result = from_dataframe(interchange_obj)

    for i in range(len(result)):
        if pd.isna(df['col'].iloc[i]):
            assert pd.isna(result['col'].iloc[i])
        else:
            assert df['col'].iloc[i] == result['col'].iloc[i]
```

**Failing input**: `categories = ['A'], codes = [-1]`

## Reproducing the Bug

```python
import pandas as pd
from pandas.api.interchange import from_dataframe

cat = pd.Categorical.from_codes([-1], categories=['A'])
df = pd.DataFrame({'col': cat})

print(f"Original value is null: {pd.isna(df['col'].iloc[0])}")

interchange_obj = df.__dataframe__()
result = from_dataframe(interchange_obj)

print(f"Result value is null: {pd.isna(result['col'].iloc[0])}")
print(f"Result value: {result['col'].iloc[0]}")
```

## Why This Is A Bug

In pandas categoricals, code -1 is the standard sentinel value for null/NA. The interchange protocol should preserve null values during conversion. Instead, the modulo operation `categories[codes % len(categories)]` at line 254 of `from_dataframe.py` wraps -1 to index 0, converting nulls to the first category. This is silent data corruption.

The comment at lines 251-252 states the modulo is to "not get ``IndexError`` for out-of-bounds sentinel values", but -1 is not just out-of-bounds—it's the designated null sentinel that must be handled specially.

## Fix

The fix should check for -1 codes before indexing, preserving them as null. The `set_nulls` call at line 263 cannot fix this because the Categorical is already constructed with wrong values.

```diff
--- a/pandas/core/interchange/from_dataframe.py
+++ b/pandas/core/interchange/from_dataframe.py
@@ -248,10 +248,12 @@ def categorical_column_to_series(col: Column) -> tuple[pd.Series, Any]:
         codes_buff, codes_dtype, offset=col.offset, length=col.size()
     )

-    # Doing module in order to not get ``IndexError`` for
-    # out-of-bounds sentinel values in `codes`
     if len(categories) > 0:
-        values = categories[codes % len(categories)]
+        # Map codes to categories, but preserve -1 (null sentinel) as None
+        # Using np.where to avoid modulo wrapping -1 to index 0
+        import numpy as np
+        values = np.where(codes == -1, None, categories[np.where(codes >= 0, codes % len(categories), 0)])
+        values = values.tolist()
     else:
         values = codes