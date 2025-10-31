# Bug Report: pandas.api.interchange.from_dataframe Categorical Null Value Corruption

**Target**: `pandas.api.interchange.from_dataframe`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When converting a pandas DataFrame with categorical columns containing null values through the interchange protocol, null values are incorrectly converted to valid category values instead of being preserved as NaN, resulting in silent data corruption.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st, settings


@given(st.lists(st.integers(min_value=-1, max_value=2), min_size=5, max_size=20))
@settings(max_examples=100)
def test_categorical_null_preservation(codes):
    categories = ['a', 'b', 'c']
    df = pd.DataFrame({'cat': pd.Categorical.from_codes(codes, categories=categories)})

    interchange_obj = df.__dataframe__()
    result = pd.api.interchange.from_dataframe(interchange_obj)

    assert result['cat'].isna().sum() == df['cat'].isna().sum(), \
        f"Null count mismatch: {result['cat'].isna().sum()} != {df['cat'].isna().sum()}"
```

<details>

<summary>
**Failing input**: `codes=[0, 0, 0, 0, -1]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 19, in <module>
    test_categorical_null_preservation()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 6, in test_categorical_null_preservation
    @settings(max_examples=100)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 14, in test_categorical_null_preservation
    assert result['cat'].isna().sum() == df['cat'].isna().sum(), \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Null count mismatch: 0 != 1
Falsifying example: test_categorical_null_preservation(
    codes=[0, 0, 0, 0, -1],
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/13/hypo.py:15
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import numpy as np

categories = ['a', 'b', 'c']
codes = [0, -1, 2]
df = pd.DataFrame({'cat': pd.Categorical.from_codes(codes, categories=categories)})

print("Original DataFrame:")
print(df)
print(f"Original values: {list(df['cat'])}")
print(f"Null values: {df['cat'].isna().sum()}")

interchange_obj = df.__dataframe__()
result = pd.api.interchange.from_dataframe(interchange_obj)

print("\nResult after round-trip:")
print(result)
print(f"Result values: {list(result['cat'])}")
print(f"Null values: {result['cat'].isna().sum()}")

print("\nAssertion check:")
print(f"Expected: {list(df['cat'])}")
print(f"Got: {list(result['cat'])}")
assert list(df['cat']) == list(result['cat']), f"Values don't match!"
```

<details>

<summary>
AssertionError: Values don't match! (NaN converted to 'c')
</summary>
```
Original DataFrame:
   cat
0    a
1  NaN
2    c
Original values: ['a', nan, 'c']
Null values: 1

Result after round-trip:
  cat
0   a
1   c
2   c
Result values: ['a', 'c', 'c']
Null values: 0

Assertion check:
Expected: ['a', nan, 'c']
Got: ['a', 'c', 'c']
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/13/repo.py", line 24, in <module>
    assert list(df['cat']) == list(result['cat']), f"Values don't match!"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Values don't match!
```
</details>

## Why This Is A Bug

This bug violates pandas' documented categorical data specification where the code `-1` explicitly represents a null/missing value. The issue occurs in the `categorical_column_to_series` function at line 254 of `from_dataframe.py`:

```python
values = categories[codes % len(categories)]
```

When processing categorical data with null values (code = -1), the modulo operation incorrectly wraps the -1 index: `-1 % 3 = 2`, causing the null value to be mapped to `categories[2]` instead of being preserved as NaN. This results in **silent data corruption** where missing values are converted to valid category values without any warning or error.

The pandas documentation clearly states that -1 is the sentinel value for missing data in categorical arrays. The interchange protocol specification also requires proper preservation of missing values during data exchange. The current implementation violates both these requirements, potentially leading to incorrect analysis results and decisions based on corrupted data.

## Relevant Context

The bug is located in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/interchange/from_dataframe.py` at lines 253-256. The comment at line 251-252 states "Doing module in order to not get IndexError for out-of-bounds sentinel values" which indicates a misunderstanding of the categorical encoding specification. The -1 sentinel is not an "out-of-bounds" value that needs wrapping - it's a special value with specific meaning (null/missing) that must be preserved.

The pandas documentation for `Categorical.from_codes` explicitly states that -1 is used to indicate missing values. This is a fundamental part of the categorical data type contract that the interchange protocol implementation must respect.

## Proposed Fix

```diff
--- a/pandas/core/interchange/from_dataframe.py
+++ b/pandas/core/interchange/from_dataframe.py
@@ -249,10 +249,17 @@ def categorical_column_to_series(col: Column) -> tuple[pd.Series, Any]:
         codes_buff, codes_dtype, offset=col.offset, length=col.size()
     )

-    # Doing module in order to not get ``IndexError`` for
-    # out-of-bounds sentinel values in `codes`
+    # Handle null values (-1 codes) properly
     if len(categories) > 0:
-        values = categories[codes % len(categories)]
+        # Create a mask for null values (code == -1)
+        null_mask = codes == -1
+        # For valid codes, map to categories; for null codes, use placeholder
+        valid_codes = np.where(null_mask, 0, codes)
+        values = categories[valid_codes]
+        # Convert to object array to allow NaN assignment
+        values = values.astype(object)
+        # Set null positions to NaN
+        values[null_mask] = np.nan
     else:
         values = codes
```