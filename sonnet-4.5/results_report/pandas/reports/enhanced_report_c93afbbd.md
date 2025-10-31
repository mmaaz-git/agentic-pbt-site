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

if __name__ == "__main__":
    test_categorical_negative_sentinel_preserved()
```

<details>

<summary>
**Failing input**: `n_categories=2, n_rows=1, codes=[-1]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 35, in <module>
    test_categorical_negative_sentinel_preserved()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 7, in test_categorical_negative_sentinel_preserved
    @settings(max_examples=100)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 28, in test_categorical_negative_sentinel_preserved
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
print(f"Original values: {df['cat_col'].values}")

xchg = df.__dataframe__()
result = pd.api.interchange.from_dataframe(xchg)

print(f"Result null positions: {np.where(result['cat_col'].isna())[0]}")
print(f"Result values: {result['cat_col'].values}")

assert np.array_equal(df['cat_col'].isna().values, result['cat_col'].isna().values), \
    "Null positions changed during interchange!"
```

<details>

<summary>
AssertionError: Null positions changed during interchange!
</summary>
```
Original null positions: [0 3]
Original values: [NaN, 'A', 'B', NaN, 'C']
Categories (3, object): ['A', 'B', 'C']
Result null positions: []
Result values: ['C', 'A', 'B', 'C', 'C']
Categories (3, object): ['A', 'B', 'C']
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/43/repo.py", line 19, in <module>
    assert np.array_equal(df['cat_col'].isna().values, result['cat_col'].isna().values), \
           ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Null positions changed during interchange!
```
</details>

## Why This Is A Bug

This violates pandas' interchange protocol specification and causes silent data corruption. The pandas documentation explicitly states that categorical null values use -1 as the sentinel value in the codes array. Specifically:

1. **Protocol Violation**: The interchange protocol explicitly defines that categorical columns use `ColumnNullType.USE_SENTINEL` with sentinel value -1 for null values. This is documented in `pandas/core/interchange/column.py:58-60`:
   ```python
   # Null values for categoricals are stored as `-1` sentinel values
   # in the category date (e.g., `col.values.codes` is int8 np.ndarray)
   DtypeKind.CATEGORICAL: (ColumnNullType.USE_SENTINEL, -1),
   ```

2. **Silent Data Corruption**: The modulo operation on line 254 of `from_dataframe.py` wraps negative sentinel values into the valid category range:
   - Sentinel value -1 becomes the last valid category index: `-1 % 3 = 2` (for 3 categories)
   - In the example above, the null values at positions [0, 3] are incorrectly mapped to category 'C'
   - Users receive no warning or error about this data corruption

3. **Unrecoverable Data Loss**: After the modulo operation, the `set_nulls()` function cannot identify the original sentinel values because:
   - The codes have already been converted to actual category values
   - When `set_nulls()` compares categorical values to the integer sentinel (-1), they never match
   - The null information is permanently lost and cannot be recovered

4. **Documentation Contradiction**: The pandas documentation explicitly states that "When working with the Categorical's codes, missing values will always have a code of -1" (pandas user guide on categorical data). The current implementation violates this fundamental specification.

## Relevant Context

The bug occurs in `pandas/core/interchange/from_dataframe.py` at lines 251-256. The code comment acknowledges that sentinel values exist in the codes array, but attempts to "fix" potential IndexErrors by using modulo:

```python
# Doing module in order to not get ``IndexError`` for
# out-of-bounds sentinel values in `codes`
if len(categories) > 0:
    values = categories[codes % len(categories)]
else:
    values = codes
```

This modulo operation is the root cause - it wraps the -1 sentinel values into the valid category range instead of preserving them as nulls. The interchange protocol specification requires proper handling of these sentinel values to maintain data integrity during interchange operations.

Key references:
- pandas documentation on categorical data: https://pandas.pydata.org/docs/user_guide/categorical.html
- DataFrame interchange protocol specification: https://data-apis.org/dataframe-protocol/latest/API.html
- Source code location: `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/interchange/from_dataframe.py:251-256`

## Proposed Fix

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