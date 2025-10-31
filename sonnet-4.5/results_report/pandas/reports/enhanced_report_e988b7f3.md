# Bug Report: pandas.core.interchange.from_dataframe Categorical Missing Value Corruption

**Target**: `pandas.core.interchange.from_dataframe.categorical_column_to_series`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `categorical_column_to_series` function silently corrupts missing values in categorical data by applying modulo arithmetic to sentinel values (-1), converting them into valid category values instead of preserving them as NaN.

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

# Run the test
test_categorical_roundtrip_preserves_nulls()
```

<details>

<summary>
**Failing input**: `n_categories=1, codes=[-1]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 34, in <module>
    test_categorical_roundtrip_preserves_nulls()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 8, in test_categorical_roundtrip_preserves_nulls
    @given(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 30, in test_categorical_roundtrip_preserves_nulls
    assert pd.isna(df_result['col'].iloc[i]), \
           ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected NaN at index 0, got 'cat0'
Falsifying example: test_categorical_roundtrip_preserves_nulls(
    n_categories=1,  # or any other generated value
    codes=[-1],
)
```
</details>

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
print(f"Expected NaN at index 2, but got: '{df_result['col'].iloc[2]}'")
```

<details>

<summary>
Missing value becomes valid category 'c'
</summary>
```
Original: ['a', 'b', NaN, 'c']
Categories (3, object): ['a', 'b', 'c']
Is null at index 2? True
Result: ['a', 'b', 'c', 'c']
Categories (3, object): ['a', 'b', 'c']
Is null at index 2? False
Expected NaN at index 2, but got: 'c'
```
</details>

## Why This Is A Bug

This violates pandas' documented behavior where -1 is the standard sentinel value for missing categorical data. According to pandas.Categorical.from_codes documentation, "codes are an integer array, where each integer points to a category... or else is -1 for NaN".

The bug occurs at line 254 in `/pandas/core/interchange/from_dataframe.py` where `values = categories[codes % len(categories)]` is executed. The modulo operation transforms -1 into a valid index: with 3 categories, -1 % 3 = 2, pointing to the third category. This silently corrupts data during the interchange protocol roundtrip, violating the fundamental principle that data should be preserved.

The existing comment at lines 251-252 acknowledges the existence of sentinel values but implements an incorrect solution that causes data corruption rather than proper null handling.

## Relevant Context

The DataFrame Interchange Protocol is designed to enable zero-copy data exchange between different DataFrame libraries. It includes support for various null representations including sentinel values through the ColumnNullType.USE_SENTINEL option. The protocol has a describe_null() method specifically to communicate how nulls are encoded.

The bug affects any categorical data with missing values when using the interchange protocol, which is particularly problematic as missing values are common in real-world datasets. The silent nature of the corruption makes it especially dangerous as users may not notice their missing values have been converted to valid categories.

Documentation:
- pandas.Categorical.from_codes: https://pandas.pydata.org/docs/reference/api/pandas.Categorical.from_codes.html
- Source code: pandas/core/interchange/from_dataframe.py:251-256

## Proposed Fix

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