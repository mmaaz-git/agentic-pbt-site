# Bug Report: pandas.api.interchange Categorical Null Values Silently Corrupted

**Target**: `pandas.api.interchange.from_dataframe`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The DataFrame interchange protocol silently corrupts null values in categorical data by converting them to actual category values. This happens because a modulo operation incorrectly maps the null sentinel value (-1) to valid category indices.

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
            assert pd.isna(result['col'].iloc[i]), f"Null at index {i} not preserved. Original: {df['col'].iloc[i]}, Result: {result['col'].iloc[i]}"
        else:
            assert df['col'].iloc[i] == result['col'].iloc[i], f"Value mismatch at index {i}. Original: {df['col'].iloc[i]}, Result: {result['col'].iloc[i]}"

if __name__ == "__main__":
    test_categorical_with_nulls()
```

<details>

<summary>
**Failing input**: `categories = ['A'], codes = [-1]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 35, in <module>
    test_categorical_with_nulls()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 6, in test_categorical_with_nulls
    @settings(max_examples=200)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 30, in test_categorical_with_nulls
    assert pd.isna(result['col'].iloc[i]), f"Null at index {i} not preserved. Original: {df['col'].iloc[i]}, Result: {result['col'].iloc[i]}"
           ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Null at index 0 not preserved. Original: nan, Result: A
Falsifying example: test_categorical_with_nulls(
    data=data(...),
)
Draw 1: 1
Draw 2: ['A']
Draw 3: [-1]
```
</details>

## Reproducing the Bug

```python
import pandas as pd
from pandas.api.interchange import from_dataframe

# Test Case 1: Single null value
print("=== Test Case 1: Single null value ===")
cat = pd.Categorical.from_codes([-1], categories=['A'])
df = pd.DataFrame({'col': cat})

print(f"Original DataFrame:")
print(df)
print(f"Original value is null: {pd.isna(df['col'].iloc[0])}")

interchange_obj = df.__dataframe__()
result = from_dataframe(interchange_obj)

print(f"\nAfter interchange conversion:")
print(result)
print(f"Result value is null: {pd.isna(result['col'].iloc[0])}")
print(f"Result value: {result['col'].iloc[0]}")

# Test Case 2: Multiple nulls with 3 categories
print("\n=== Test Case 2: Multiple nulls (3 categories) ===")
cat2 = pd.Categorical.from_codes([-1, 0, -1, 1, -1], categories=['A', 'B', 'C'])
df2 = pd.DataFrame({'col': cat2})

print(f"Original DataFrame:")
print(df2)
print(f"Original nulls at indices: {[i for i in range(len(df2)) if pd.isna(df2['col'].iloc[i])]}")

interchange_obj2 = df2.__dataframe__()
result2 = from_dataframe(interchange_obj2)

print(f"\nAfter interchange conversion:")
print(result2)
print(f"Result nulls at indices: {[i for i in range(len(result2)) if pd.isna(result2['col'].iloc[i])]}")

# Show what -1 % 3 equals
print(f"\n-1 % 3 = {-1 % 3}")
print(f"categories[2] = '{['A', 'B', 'C'][2]}'")
```

<details>

<summary>
Silent data corruption: NaN values become valid categories
</summary>
```
=== Test Case 1: Single null value ===
Original DataFrame:
   col
0  NaN
Original value is null: True

After interchange conversion:
  col
0   A
Result value is null: False
Result value: A

=== Test Case 2: Multiple nulls (3 categories) ===
Original DataFrame:
   col
0  NaN
1    A
2  NaN
3    B
4  NaN
Original nulls at indices: [0, 2, 4]

After interchange conversion:
  col
0   C
1   A
2   C
3   B
4   C
Result nulls at indices: []

-1 % 3 = 2
categories[2] = 'C'
```
</details>

## Why This Is A Bug

This is a critical data integrity violation where null/missing values are silently converted to actual data values. In pandas categoricals, the code -1 is the well-documented sentinel value representing null/NA, as stated in the official pandas documentation: "When working with the Categorical's codes, missing values will always have a code of -1."

The bug occurs at line 254 of `/pandas/core/interchange/from_dataframe.py`:
```python
values = categories[codes % len(categories)]
```

This modulo operation wraps the -1 sentinel to a valid category index:
- With 1 category: `-1 % 1 = 0` → maps to `categories[0]`
- With 3 categories: `-1 % 3 = 2` → maps to `categories[2]`

The comment at lines 251-252 acknowledges "out-of-bounds sentinel values" but misunderstands their purpose—they're not just out-of-bounds values to avoid IndexError, but specific null indicators that must be preserved. The `set_nulls` function called later (line 263) cannot fix this because the Categorical is already constructed with incorrect values.

This violates the fundamental contract of data interchange: preserving data semantics. Converting missing data to valid values without warning is among the most serious types of bugs in data processing, as it can lead to incorrect analysis results and wrong business decisions.

## Relevant Context

The pandas documentation explicitly states that the DataFrame interchange protocol has "severe implementation issues" and recommends using the Arrow C Data Interface for pandas 2.3+. However, this doesn't excuse silent data corruption in a published API that users may depend on.

Key documentation references:
- Pandas Categorical documentation confirms -1 as the null sentinel: https://pandas.pydata.org/docs/user_guide/categorical.html
- DataFrame Interchange Protocol specification: https://data-apis.org/dataframe-protocol/latest/
- Source code location: `/pandas/core/interchange/from_dataframe.py:254`

The bug is 100% reproducible and deterministic. Any categorical data with null values will be corrupted when passed through the interchange protocol.

## Proposed Fix

```diff
--- a/pandas/core/interchange/from_dataframe.py
+++ b/pandas/core/interchange/from_dataframe.py
@@ -248,11 +248,16 @@ def categorical_column_to_series(col: Column) -> tuple[pd.Series, Any]:
         codes_buff, codes_dtype, offset=col.offset, length=col.size()
     )

-    # Doing module in order to not get ``IndexError`` for
-    # out-of-bounds sentinel values in `codes`
+    # Handle null sentinel values (-1) correctly
     if len(categories) > 0:
-        values = categories[codes % len(categories)]
+        # Create an array to hold the values, preserving nulls
+        import numpy as np
+        values = np.empty(len(codes), dtype=object)
+        for i, code in enumerate(codes):
+            if code == -1:  # Null sentinel
+                values[i] = None
+            else:
+                values[i] = categories[code % len(categories)]
     else:
         values = codes
```