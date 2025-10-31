# Bug Report: pandas.api.interchange Categorical Nulls Silently Corrupted During Round-Trip

**Target**: `pandas.api.interchange.from_dataframe`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Null values in categorical columns are silently converted to valid category values when round-tripping through the DataFrame interchange protocol, causing data corruption without any warnings or errors.

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

<details>

<summary>
**Failing input**: `cat_values=['a', 'a'], null_mask=[False, True]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 22, in <module>
    test_from_dataframe_categorical_with_nulls()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 7, in test_from_dataframe_categorical_with_nulls
    st.lists(st.booleans(), min_size=1, max_size=50))
            ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 18, in test_from_dataframe_categorical_with_nulls
    pd.testing.assert_frame_equal(result.reset_index(drop=True), df.reset_index(drop=True), check_dtype=False, check_categorical=False)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 1303, in assert_frame_equal
    assert_series_equal(
    ~~~~~~~~~~~~~~~~~~~^
        lcol,
        ^^^^^
    ...<12 lines>...
        check_flags=False,
        ^^^^^^^^^^^^^^^^^^
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
AssertionError: DataFrame.iloc[:, 0] (column name="cat") are different

DataFrame.iloc[:, 0] (column name="cat") values are different (50.0 %)
[index]: [0, 1]
[left]:  ['a', 'a']
Categories (1, object): ['a']
[right]: ['a', NaN]
Categories (1, object): ['a']
At positional index 1, first diff: a != nan
Falsifying example: test_from_dataframe_categorical_with_nulls(
    cat_values=['a', 'a'],
    null_mask=[False, True],
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import pandas.api.interchange as interchange

# Create a DataFrame with a categorical column containing null values
df = pd.DataFrame({'cat': pd.Categorical(['a', None])})
print("Original DataFrame:")
print(df)
print(f"Type of original column: {type(df['cat'])}")
print(f"Original categories: {df['cat'].cat.categories.tolist()}")
print(f"Original codes: {df['cat'].cat.codes.tolist()}")

# Round-trip through interchange protocol
interchange_obj = df.__dataframe__()
result = interchange.from_dataframe(interchange_obj)

print("\nResult DataFrame after round-trip:")
print(result)
print(f"Type of result column: {type(result['cat'])}")
print(f"Result categories: {result['cat'].cat.categories.tolist()}")
print(f"Result codes: {result['cat'].cat.codes.tolist()}")

# Check if null is preserved
print(f"\nOriginal value at position 1: {df.iloc[1, 0]}")
print(f"Result value at position 1: {result.iloc[1, 0]}")
print(f"Is original value null? {pd.isna(df.iloc[1, 0])}")
print(f"Is result value null? {pd.isna(result.iloc[1, 0])}")

# This assertion should pass but will fail due to the bug
try:
    assert pd.isna(result.iloc[1, 0]), f"Expected null but got {result.iloc[1, 0]}"
    print("\n✓ Assertion passed: Null value was preserved")
except AssertionError as e:
    print(f"\n✗ Assertion failed: {e}")
```

<details>

<summary>
Silent data corruption: NaN becomes 'a'
</summary>
```
Original DataFrame:
   cat
0    a
1  NaN
Type of original column: <class 'pandas.core.series.Series'>
Original categories: ['a']
Original codes: [0, -1]

Result DataFrame after round-trip:
  cat
0   a
1   a
Type of result column: <class 'pandas.core.series.Series'>
Result categories: ['a']
Result codes: [0, 0]

Original value at position 1: nan
Result value at position 1: a
Is original value null? True
Is result value null? False

✗ Assertion failed: Expected null but got a
```
</details>

## Why This Is A Bug

This bug violates the fundamental contract of the DataFrame interchange protocol - data must be preserved when converting between libraries. The issue causes **silent data corruption** where missing values are replaced with valid category values without any warnings or errors.

The bug occurs because pandas uses sentinel values (e.g., -1) to represent null values in categorical codes. When the `categorical_column_to_series` function in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/interchange/from_dataframe.py` processes these codes at line 254, it uses a modulo operation:

```python
values = categories[codes % len(categories)]
```

This modulo operation was intended to avoid IndexError for out-of-bounds sentinel values, but it inadvertently maps these sentinel values to valid category indices. For example, with one category: `-1 % 1 = 0`, which maps the null to the first category ('a').

While `set_nulls` is called afterwards (line 263), it cannot properly restore the null values because the Categorical has already been created with incorrect values, and the sentinel information has been lost during the modulo operation.

This is particularly problematic because:
1. **Data integrity is compromised** - Users lose critical information about missing data
2. **The corruption is silent** - No warnings or errors alert users to the data loss
3. **It's inconsistent** - Other data types (numeric, string) preserve nulls correctly through the same protocol
4. **It affects common use cases** - Categorical data with nulls is extremely common in real-world datasets

## Relevant Context

The DataFrame interchange protocol is designed to facilitate data exchange between different dataframe libraries (pandas, polars, cuDF, etc.). The protocol explicitly supports multiple null representation types including `USE_SENTINEL`, which is what pandas uses for categorical columns.

The code comment at lines 251-252 acknowledges that sentinel values exist:
```python
# Doing module in order to not get ``IndexError`` for
# out-of-bounds sentinel values in `codes`
```

This shows the developer was aware of sentinel values but the chosen solution (modulo operation) inadvertently causes data corruption.

For reference:
- Source file: `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/interchange/from_dataframe.py`
- Bug location: Line 254 in `categorical_column_to_series` function
- pandas documentation: https://pandas.pydata.org/docs/reference/api/pandas.api.interchange.from_dataframe.html
- Interchange protocol spec: https://data-apis.org/dataframe-protocol/latest/

## Proposed Fix

```diff
--- a/pandas/core/interchange/from_dataframe.py
+++ b/pandas/core/interchange/from_dataframe.py
@@ -248,12 +248,17 @@ def categorical_column_to_series(col: Column) -> tuple[pd.Series, Any]:
         codes_buff, codes_dtype, offset=col.offset, length=col.size()
     )

-    # Doing module in order to not get ``IndexError`` for
-    # out-of-bounds sentinel values in `codes`
+    # Handle out-of-bounds codes (which represent nulls) before indexing
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