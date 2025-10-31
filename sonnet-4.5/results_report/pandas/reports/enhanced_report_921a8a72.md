# Bug Report: pandas.core.interchange.from_dataframe Categorical Missing Value Corruption

**Target**: `pandas.core.interchange.from_dataframe.categorical_column_to_series`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The pandas interchange protocol silently corrupts categorical data containing missing values by converting NaN values into valid category values during data exchange.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, Verbosity
import pandas as pd

@given(st.lists(st.text(min_size=1, max_size=5), min_size=1, max_size=10))
@settings(verbosity=Verbosity.verbose, max_examples=10)
def test_categorical_preserves_missing_values(categories):
    """Categorical -1 codes (missing values) should round-trip as NaN."""
    cat_data = pd.Categorical.from_codes([-1], categories=list(set(categories)), ordered=False)
    series = pd.Series(cat_data)

    from pandas.core.interchange.from_dataframe import from_dataframe
    df = pd.DataFrame({"cat": series})
    result = from_dataframe(df.__dataframe__(allow_copy=True))

    assert pd.isna(result["cat"].iloc[0]), f"Missing categorical value not preserved for categories={list(set(categories))}"

if __name__ == "__main__":
    test_categorical_preserves_missing_values()
```

<details>

<summary>
**Failing input**: `categories=['0']`
</summary>
```
Trying example: test_categorical_preserves_missing_values(
    categories=['0'],
)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 15, in test_categorical_preserves_missing_values
    assert pd.isna(result["cat"].iloc[0]), f"Missing categorical value not preserved for categories={list(set(categories))}"
           ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Missing categorical value not preserved for categories=['0']

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 18, in <module>
    test_categorical_preserves_missing_values()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 5, in test_categorical_preserves_missing_values
    @settings(verbosity=Verbosity.verbose, max_examples=10)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 15, in test_categorical_preserves_missing_values
    assert pd.isna(result["cat"].iloc[0]), f"Missing categorical value not preserved for categories={list(set(categories))}"
           ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Missing categorical value not preserved for categories=['0']
Falsifying example: test_categorical_preserves_missing_values(
    categories=['0'],
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
from pandas.core.interchange.from_dataframe import categorical_column_to_series

# Create categorical data with missing value (-1 is the sentinel for missing)
cat_data = pd.Categorical.from_codes([-1], categories=['a'], ordered=False)
series = pd.Series(cat_data, name="cat_col")

print(f"Original series value: {series.iloc[0]}")
print(f"Original is NaN: {pd.isna(series.iloc[0])}")

# Create interchange object and extract column
interchange_obj = pd.DataFrame({"cat_col": series}).__dataframe__(allow_copy=True)
col = interchange_obj.get_column_by_name("cat_col")

# Convert using the function that has the bug
result_series, _ = categorical_column_to_series(col)

print(f"After interchange value: {result_series.iloc[0]}")
print(f"After interchange is NaN: {pd.isna(result_series.iloc[0])}")

# Show the problem: NaN became a valid category
if pd.isna(series.iloc[0]) and not pd.isna(result_series.iloc[0]):
    print("\nBUG CONFIRMED: Missing value (NaN) was converted to a valid category!")
```

<details>

<summary>
Output shows NaN corruption to valid category
</summary>
```
Original series value: nan
Original is NaN: True
After interchange value: a
After interchange is NaN: False

BUG CONFIRMED: Missing value (NaN) was converted to a valid category!
```
</details>

## Why This Is A Bug

In pandas, categorical code -1 is the well-documented sentinel value for missing data in categorical arrays. The pandas documentation explicitly states that -1 represents missing values in categorical codes. The interchange protocol's column metadata correctly identifies -1 as the sentinel value (USE_SENTINEL with value -1).

The bug occurs in `categorical_column_to_series` at line 254 of `/pandas/core/interchange/from_dataframe.py`:
```python
values = categories[codes % len(categories)]  # -1 % 1 = 0!
```

This modulo operation transforms -1 codes to valid array indices before the `set_nulls` function can identify them as missing values. When there's only one category, `-1 % 1 = 0`, mapping the missing value to the first (and only) category.

The function `set_nulls` expects to see -1 sentinel values in the data to convert them to NaN (line 527: `null_pos = pd.Series(data) == sentinel_val`), but it never gets the chance because the modulo operation has already converted -1 to a valid category value.

This violates fundamental data preservation principles - missing values should never silently become valid data during interchange operations.

## Relevant Context

The pandas interchange protocol documentation includes warnings about "severe implementation issues" and recommends using Arrow C Data Interface instead (pandas 2.3+). However, as long as the interchange protocol remains part of the public API, data corruption bugs should be fixed.

The comment at line 251-252 claims the modulo is used "in order to not get ``IndexError`` for out-of-bounds sentinel values in `codes`", but this creates a far worse problem than an IndexError - silent data corruption.

Relevant documentation:
- [Pandas Categorical Documentation](https://pandas.pydata.org/docs/reference/api/pandas.Categorical.html) - States that -1 is the missing value sentinel
- [Interchange Protocol Spec](https://data-apis.org/dataframe-protocol/latest/) - Defines USE_SENTINEL for missing value handling

## Proposed Fix

The issue can be fixed by preserving -1 codes as -1 values (or another sentinel) so that `set_nulls` can properly identify and convert them to NaN:

```diff
--- a/pandas/core/interchange/from_dataframe.py
+++ b/pandas/core/interchange/from_dataframe.py
@@ -250,10 +250,13 @@ def categorical_column_to_series(col: Column) -> tuple[pd.Series, Any]:

-    # Doing module in order to not get ``IndexError`` for
-    # out-of-bounds sentinel values in `codes`
+    # Map codes to categories, preserving -1 (missing value sentinel)
+    # -1 codes will be handled by set_nulls below
     if len(categories) > 0:
-        values = categories[codes % len(categories)]
+        valid_mask = codes >= 0
+        values = np.empty(len(codes), dtype=object)
+        values[valid_mask] = categories[codes[valid_mask]]
+        values[~valid_mask] = -1
     else:
         values = codes
```