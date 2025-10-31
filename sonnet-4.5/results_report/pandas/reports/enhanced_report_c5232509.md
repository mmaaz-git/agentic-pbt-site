# Bug Report: pandas.api.interchange Categorical Null Values Lost in Round-Trip

**Target**: `pandas.core.interchange.from_dataframe.categorical_column_to_series`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Null values in categorical columns are silently converted to valid category values when using the DataFrame interchange protocol round-trip conversion, causing data loss without any warning or error.

## Property-Based Test

```python
from hypothesis import given, strategies as st, example
import pandas as pd
from pandas.api.interchange import from_dataframe


@given(
    st.lists(st.sampled_from(["cat1", "cat2", "cat3", None]), min_size=0, max_size=100),
    st.booleans()
)
@example(['cat1', None], False)  # Add the specific failing case
def test_round_trip_categorical(cat_list, ordered):
    df = pd.DataFrame({"col": pd.Categorical(cat_list, ordered=ordered)})
    result = from_dataframe(df.__dataframe__())
    pd.testing.assert_frame_equal(result, df)


# Run the test
if __name__ == "__main__":
    print("Running property-based tests...")
    # Run the property-based test
    test_round_trip_categorical()
```

<details>

<summary>
**Failing input**: `cat_list=['cat1', None], ordered=False`
</summary>
```
Running property-based tests...
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 21, in <module>
    test_round_trip_categorical()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 7, in test_round_trip_categorical
    st.lists(st.sampled_from(["cat1", "cat2", "cat3", None]), min_size=0, max_size=100),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 14, in test_round_trip_categorical
    pd.testing.assert_frame_equal(result, df)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^
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
AssertionError: DataFrame.iloc[:, 0] (column name="col") are different

DataFrame.iloc[:, 0] (column name="col") values are different (50.0 %)
[index]: [0, 1]
[left]:  ['cat1', 'cat1']
Categories (1, object): ['cat1']
[right]: ['cat1', NaN]
Categories (1, object): ['cat1']
At positional index 1, first diff: cat1 != nan
Falsifying explicit example: test_round_trip_categorical(
    cat_list=['cat1', None],
    ordered=False,
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
from pandas.api.interchange import from_dataframe

# Create a DataFrame with categorical column containing null
df = pd.DataFrame({"col": pd.Categorical(["cat1", None])})
print("Original DataFrame:")
print(df)
print("\nOriginal values as list:")
print(df["col"].tolist())

# Convert through the interchange protocol
result = from_dataframe(df.__dataframe__())
print("\nDataFrame after round-trip:")
print(result)
print("\nValues after round-trip as list:")
print(result["col"].tolist())

# Check if values match
print("\nComparison:")
print(f"Original: {df['col'].tolist()}")
print(f"After round-trip: {result['col'].tolist()}")
print(f"Are they equal? {df['col'].tolist() == result['col'].tolist()}")
```

<details>

<summary>
Output showing null value converted to 'cat1'
</summary>
```
Original DataFrame:
    col
0  cat1
1   NaN

Original values as list:
['cat1', nan]

DataFrame after round-trip:
    col
0  cat1
1  cat1

Values after round-trip as list:
['cat1', 'cat1']

Comparison:
Original: ['cat1', nan]
After round-trip: ['cat1', 'cat1']
Are they equal? False
```
</details>

## Why This Is A Bug

The DataFrame interchange protocol is designed to preserve data integrity during conversion between different dataframe libraries. The protocol explicitly defines how null values should be represented in categorical columns using sentinel values (ColumnNullType.USE_SENTINEL with value -1).

The bug violates this expected behavior in multiple ways:

1. **Protocol Violation**: The interchange protocol specification defines USE_SENTINEL for representing null values in categorical columns. The implementation acknowledges this (see comment at line 251-252 of from_dataframe.py) but fails to preserve these sentinel values.

2. **Silent Data Corruption**: Null values are silently converted to valid category values without any warning or error. This is particularly dangerous because users may not immediately notice their missing data has been replaced with incorrect values.

3. **Implementation Intent**: The code comment at lines 251-252 explicitly states "Doing module in order to not get ``IndexError`` for out-of-bounds sentinel values in `codes`", demonstrating clear intent to handle sentinel values. However, the modulo operation inadvertently maps sentinel values to valid category indices.

4. **Incorrect Logic**: When a null is represented by sentinel value -1, the modulo operation `codes % len(categories)` produces:
   - For 2 categories: -1 % 2 = 1 (maps to second category)
   - For 3 categories: -1 % 3 = 2 (maps to third category)
   - This systematically corrupts all null values

5. **Function Ordering Issue**: The `set_nulls()` function is called after the incorrect conversion has already happened (line 263), meaning it cannot recover the lost information about which values were originally null.

## Relevant Context

The pandas documentation provides important context:

1. **Alternative Recommended**: From pandas 2.3 onwards, the documentation recommends using Arrow PyCapsule Interface instead of the interchange protocol, noting "severe implementation issues" with the interchange protocol.

2. **Known Issues**: The documentation explicitly warns about implementation problems and recommends limited use cases for the interchange protocol.

3. **Code Location**: The bug is in `/pandas/core/interchange/from_dataframe.py`, specifically in the `categorical_column_to_series()` function at lines 253-254.

4. **Impact Scope**: This affects any categorical column with null values being converted through the interchange protocol, regardless of whether the categorical is ordered or unordered.

Relevant documentation links:
- DataFrame Interchange Protocol Spec: https://data-apis.org/dataframe-protocol/latest/
- pandas.api.interchange.from_dataframe: https://pandas.pydata.org/docs/reference/api/pandas.api.interchange.from_dataframe.html

## Proposed Fix

```diff
--- a/pandas/core/interchange/from_dataframe.py
+++ b/pandas/core/interchange/from_dataframe.py
@@ -248,10 +248,21 @@ def categorical_column_to_series(col: Column) -> tuple[pd.Series, Any]:
         codes_buff, codes_dtype, offset=col.offset, length=col.size()
     )

-    # Doing module in order to not get ``IndexError`` for
-    # out-of-bounds sentinel values in `codes`
+    null_kind, sentinel_val = col.describe_null
+
+    # Create values array, handling sentinel values for nulls
     if len(categories) > 0:
-        values = categories[codes % len(categories)]
+        # Handle sentinel values properly to preserve null information
+        if null_kind == ColumnNullType.USE_SENTINEL:
+            # Create mask for valid (non-sentinel) codes
+            valid_mask = codes != sentinel_val
+            values = np.empty(len(codes), dtype=object)
+            # Only apply modulo to valid codes, not sentinel values
+            values[valid_mask] = categories[codes[valid_mask] % len(categories)]
+            # Preserve sentinel values for set_nulls to process later
+            values[~valid_mask] = sentinel_val
+        else:
+            # No sentinel values, safe to use modulo on all codes
+            values = categories[codes % len(categories)]
     else:
         values = codes
```