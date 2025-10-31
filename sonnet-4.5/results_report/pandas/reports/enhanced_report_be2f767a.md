# Bug Report: pandas.api.interchange Categorical Null Values Silently Converted to Valid Data

**Target**: `pandas.core.interchange.from_dataframe.categorical_column_to_series`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The pandas interchange protocol incorrectly converts null sentinel values (-1) in categorical columns to valid category values through a modulo operation, causing silent data corruption where missing values become actual data points.

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

<details>

<summary>
**Failing input**: `n_categories=2, n_rows=1, codes=[-1]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 36, in <module>
    test_categorical_negative_sentinel_preserved()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 7, in test_categorical_negative_sentinel_preserved
    @settings(max_examples=100)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 28, in test_categorical_negative_sentinel_preserved
    np.testing.assert_array_equal(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        df["cat_col"].isna().values,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        result["cat_col"].isna().values,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        err_msg="Null positions don't match after interchange"
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py", line 1051, in assert_array_equal
    assert_array_compare(operator.__eq__, actual, desired, err_msg=err_msg,
    ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                         verbose=verbose, header='Arrays are not equal',
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                         strict=strict)
                         ^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py", line 916, in assert_array_compare
    raise AssertionError(msg)
AssertionError:
Arrays are not equal
Null positions don't match after interchange
Mismatched elements: 1 / 1 (100%)
 ACTUAL: array([ True])
 DESIRED: array([False])
Falsifying example: test_categorical_negative_sentinel_preserved(
    data=data(...),
)
Draw 1: 2
Draw 2: 1
Draw 3: True
```
</details>

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

<details>

<summary>
AssertionError: Null positions changed during interchange!
</summary>
```
Original null positions: [0 3]
Result null positions: []
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/28/repo.py", line 17, in <module>
    assert np.array_equal(df['cat_col'].isna().values, result['cat_col'].isna().values), \
           ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Null positions changed during interchange!
```
</details>

## Why This Is A Bug

This violates the pandas interchange protocol specification and causes silent data corruption. The code explicitly documents that categorical columns use `-1` as the sentinel value for nulls (`pandas/core/interchange/column.py:60`), and the `set_nulls` function is designed to restore nulls by comparing data to sentinel values. However, the modulo operation at line 254 in `from_dataframe.py` destroys these sentinel values before `set_nulls` can process them.

When sentinel value -1 is processed with `codes % len(categories)`, it wraps around to become a valid category index. For example, with 3 categories, `-1 % 3 = 2`, mapping the null to the last category "C". This silent conversion means:
- Statistical analyses produce incorrect counts and distributions
- Filtering operations miss or wrongly include null values
- Data integrity is compromised without any warning to users
- The `set_nulls` function becomes completely ineffective for categoricals

## Relevant Context

The interchange protocol is pandas' mechanism for exchanging data with other dataframe libraries. The protocol's architecture includes:
- `_NULL_DESCRIPTION` dictionary (`column.py:52-63`) explicitly defining `-1` as categorical null sentinel
- `ColumnNullType.USE_SENTINEL` enum indicating sentinel-based null representation
- `set_nulls()` function (`from_dataframe.py:494-557`) designed to identify and restore nulls by checking `data == sentinel_val`
- Comment at line 251-252 acknowledging sentinel values exist but implementing a harmful "fix"

Documentation: https://pandas.pydata.org/docs/reference/api/pandas.api.interchange.from_dataframe.html
Source: https://github.com/pandas-dev/pandas/blob/main/pandas/core/interchange/from_dataframe.py

## Proposed Fix

The sentinel values must be handled before indexing into categories, not corrupted via modulo:

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
+        # Create values array, preserving sentinel values for null handling
+        values = np.empty(len(codes), dtype=object)
+        if null_kind == ColumnNullType.USE_SENTINEL:
+            valid_mask = codes != sentinel_val
+            values[valid_mask] = categories[codes[valid_mask]]
+            values[~valid_mask] = None
+        else:
+            values[:] = categories[codes]
     else:
         values = codes
-
+
     cat = pd.Categorical(
         values, categories=categories, ordered=categorical["is_ordered"]
     )
```