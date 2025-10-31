# Bug Report: pandas.api.interchange Categorical Null Values Lost

**Target**: `pandas.api.interchange.from_dataframe`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Null values in categorical columns are silently converted to actual category values when round-tripping through the DataFrame interchange protocol, causing data corruption without any error or warning.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd
from pandas.api.interchange import from_dataframe


@given(
    st.lists(st.sampled_from(["a", "b", "c", None]), min_size=1, max_size=50)
)
@settings(max_examples=100)
def test_categorical_with_nulls(cat_values):
    """Test that categorical columns with nulls round-trip correctly through interchange protocol."""
    df = pd.DataFrame({"cat_col": pd.Categorical(cat_values)})

    interchange_df = df.__dataframe__()
    result_df = from_dataframe(interchange_df)

    pd.testing.assert_series_equal(
        result_df["cat_col"].reset_index(drop=True),
        df["cat_col"].reset_index(drop=True),
        check_categorical=True
    )


if __name__ == "__main__":
    # Run the test
    test_categorical_with_nulls()
```

<details>

<summary>
**Failing input**: `cat_values=['a', None]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 26, in <module>
    test_categorical_with_nulls()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 7, in test_categorical_with_nulls
    st.lists(st.sampled_from(["a", "b", "c", None]), min_size=1, max_size=50)
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 17, in test_categorical_with_nulls
    pd.testing.assert_series_equal(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        result_df["cat_col"].reset_index(drop=True),
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        df["cat_col"].reset_index(drop=True),
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        check_categorical=True
        ^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 1050, in assert_series_equal
    _testing.assert_almost_equal(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        left._values,
        ^^^^^^^^^^^^^
    ...<5 lines>...
        index_values=left.index,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "pandas/_libs/testing.pyx", line 55, in pandas._libs.testing.assert_almost_equal
  File "pandas/_libs/testing.pyx", line 173, in pandas._libs.testing.assert_almost_equal
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 620, in raise_assert_detail
    raise AssertionError(msg)
AssertionError: Series are different

Series values are different (50.0 %)
[index]: [0, 1]
[left]:  ['a', 'a']
Categories (1, object): ['a']
[right]: ['a', NaN]
Categories (1, object): ['a']
At positional index 1, first diff: a != nan
Falsifying example: test_categorical_with_nulls(
    cat_values=['a', None],
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/_dtype.py:33
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:3614
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/_config/config.py:138
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/_config/config.py:628
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/_config/config.py:659
        (and 30 more with settings.verbosity >= verbose)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
from pandas.api.interchange import from_dataframe

# Create a DataFrame with categorical column containing null values
df = pd.DataFrame({"cat_col": pd.Categorical(['a', None])})
print("Original DataFrame:")
print(f"  Values: {list(df['cat_col'])}")
print(f"  Is null: {list(df['cat_col'].isna())}")

# Convert through interchange protocol
interchange_df = df.__dataframe__()
result_df = from_dataframe(interchange_df)

print("\nAfter round-trip through interchange protocol:")
print(f"  Values: {list(result_df['cat_col'])}")
print(f"  Is null: {list(result_df['cat_col'].isna())}")

# Show that null has been converted to actual category
print("\nBug demonstrated:")
print(f"  Original second value is null: {pd.isna(df['cat_col'].iloc[1])}")
print(f"  Result second value is null: {pd.isna(result_df['cat_col'].iloc[1])}")
print(f"  Result second value: '{result_df['cat_col'].iloc[1]}'")

# Additional test with multiple nulls and categories
print("\n" + "="*60)
print("Testing with multiple categories and nulls:")
df2 = pd.DataFrame({"cat_col": pd.Categorical(['a', 'b', None, 'c', None])})
print(f"Original: {list(df2['cat_col'])}")

interchange_df2 = df2.__dataframe__()
result_df2 = from_dataframe(interchange_df2)
print(f"After round-trip: {list(result_df2['cat_col'])}")
```

<details>

<summary>
Data corruption: Null values converted to category values
</summary>
```
Original DataFrame:
  Values: ['a', nan]
  Is null: [False, True]

After round-trip through interchange protocol:
  Values: ['a', 'a']
  Is null: [False, False]

Bug demonstrated:
  Original second value is null: True
  Result second value is null: False
  Result second value: 'a'

============================================================
Testing with multiple categories and nulls:
Original: ['a', 'b', nan, 'c', nan]
After round-trip: ['a', 'b', 'c', 'c', 'c']
```
</details>

## Why This Is A Bug

This bug violates the DataFrame interchange protocol specification and causes silent data corruption. The issue occurs because:

1. **Protocol Specification Violation**: The interchange protocol explicitly specifies that categorical null values should be represented as `-1` sentinel values in the codes array, as documented in `pandas/core/interchange/column.py:60`: "Null values for categoricals are stored as `-1` sentinel values".

2. **Incorrect Modulo Operation**: In `pandas/core/interchange/from_dataframe.py:254`, the code uses modulo arithmetic to prevent IndexError:
   ```python
   values = categories[codes % len(categories)]
   ```
   This causes `-1 % len(categories)` to wrap around to a valid index, converting nulls to actual category values:
   - With 1 category `['a']`: `-1 % 1 = 0` → maps to `'a'`
   - With 3 categories `['a', 'b', 'c']`: `-1 % 3 = 2` → maps to `'c'`

3. **Silent Data Corruption**: No error or warning is raised. Data is silently corrupted, which is worse than a crash as users may not notice the corruption until much later in their analysis pipeline.

4. **Developer Intent Mismatch**: The comment at lines 251-252 shows the developer was aware of sentinel values ("Doing module in order to not get IndexError for out-of-bounds sentinel values") but implemented the handling incorrectly.

## Relevant Context

- The interchange protocol documentation states that categorical columns use `ColumnNullType.USE_SENTINEL` with value `-1` for null representation
- This affects any DataFrame with categorical columns containing null values when converted through the interchange protocol
- The pandas documentation warns about "severe implementation issues" with the interchange protocol, and this bug is a critical example
- Related source files:
  - `/pandas/core/interchange/from_dataframe.py:254` (bug location)
  - `/pandas/core/interchange/column.py:60` (sentinel value specification)

## Proposed Fix

The fix should check for sentinel values before applying the modulo operation to preserve null values correctly:

```diff
--- a/pandas/core/interchange/from_dataframe.py
+++ b/pandas/core/interchange/from_dataframe.py
@@ -251,7 +251,11 @@ def categorical_column_to_series(col: Column) -> tuple[pd.Series, Any]:
     # Doing module in order to not get IndexError for
     # out-of-bounds sentinel values in `codes`
     if len(categories) > 0:
-        values = categories[codes % len(categories)]
+        # Preserve null sentinel values (-1) before modulo operation
+        null_mask = codes == -1
+        safe_codes = codes.copy()
+        safe_codes[null_mask] = 0  # Use any valid index temporarily
+        values = categories[safe_codes % len(categories)]
+        values[null_mask] = None  # Restore nulls
     else:
         values = codes
```