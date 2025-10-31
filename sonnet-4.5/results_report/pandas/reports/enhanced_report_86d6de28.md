# Bug Report: pandas.DataFrame.to_dict 'tight' Orient Computes But Never Uses Optimized Data Variable

**Target**: `pandas.DataFrame.to_dict`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `to_dict` method with 'tight' orientation computes an optimized `data` variable using the `_create_data_for_split_and_tight_to_dict` helper function but never uses it, instead recomputing the data inefficiently, resulting in 3.3x slower performance.

## Property-Based Test

```python
from hypothesis import given, settings
from hypothesis.extra.pandas import data_frames, column
from hypothesis import strategies as st
import pandas as pd


@settings(max_examples=200)
@given(data_frames([
    column('int_col', dtype=int),
    column('float_col', dtype=float),
    column('str_col', dtype=str)
], index=st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=20, unique=True)))
def test_to_dict_tight_should_use_computed_data(df):
    """Test that 'split' and 'tight' orientations produce identical data values."""
    split_result = df.to_dict(orient='split')
    tight_result = df.to_dict(orient='tight')

    assert split_result['data'] == tight_result['data'], \
        f"Data mismatch: split and tight should produce identical data values"

    # Also verify the structural differences are as expected
    assert 'index_names' in tight_result, "'tight' should include 'index_names'"
    assert 'column_names' in tight_result, "'tight' should include 'column_names'"
    assert 'index_names' not in split_result, "'split' should not include 'index_names'"
    assert 'column_names' not in split_result, "'split' should not include 'column_names'"

if __name__ == "__main__":
    # Run the test
    test_to_dict_tight_should_use_computed_data()
    print("All property-based tests passed successfully!")
```

<details>

<summary>
**Failing input**: N/A - This test passes; the bug is a performance/code quality issue, not a correctness issue
</summary>
```
All property-based tests passed successfully!
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import time

# Create a test DataFrame with mixed types
df = pd.DataFrame({
    'int_col': list(range(100000)),
    'float_col': [float(i) * 0.1 for i in range(100000)],
    'str_col': [f'value_{i}' for i in range(100000)]
})

print("Testing pandas.DataFrame.to_dict() with 'split' and 'tight' orientations")
print("=" * 70)
print(f"DataFrame shape: {df.shape}")
print(f"Column types: {df.dtypes.to_dict()}")
print()

# Test 'split' orientation
start = time.time()
split_result = df.to_dict(orient='split')
split_time = time.time() - start
print(f"'split' orientation time: {split_time:.4f} seconds")

# Test 'tight' orientation
start = time.time()
tight_result = df.to_dict(orient='tight')
tight_time = time.time() - start
print(f"'tight' orientation time: {tight_time:.4f} seconds")

print()
print(f"Performance difference: 'tight' is {tight_time/split_time:.1f}x slower than 'split'")

# Verify that the data is the same
print()
print("Data comparison:")
print(f"Data values identical: {split_result['data'] == tight_result['data']}")

# Show the structural difference between split and tight
print()
print("Keys in 'split' result:", list(split_result.keys()))
print("Keys in 'tight' result:", list(tight_result.keys()))

# Demonstrate that 'tight' has additional metadata
if 'index_names' in tight_result:
    print(f"'tight' includes index_names: {tight_result['index_names']}")
if 'column_names' in tight_result:
    print(f"'tight' includes column_names: {tight_result['column_names']}")
```

<details>

<summary>
Performance degradation demonstrating the impact of the unused optimization
</summary>
```
Testing pandas.DataFrame.to_dict() with 'split' and 'tight' orientations
======================================================================
DataFrame shape: (100000, 3)
Column types: {'int_col': dtype('int64'), 'float_col': dtype('float64'), 'str_col': dtype('O')}

'split' orientation time: 0.0334 seconds
'tight' orientation time: 0.1098 seconds

Performance difference: 'tight' is 3.3x slower than 'split'

Data comparison:
Data values identical: True

Keys in 'split' result: ['index', 'columns', 'data']
Keys in 'tight' result: ['index', 'columns', 'data', 'index_names', 'column_names']
'tight' includes index_names: [None]
'tight' includes column_names: [None]
```
</details>

## Why This Is A Bug

The 'tight' orientation implementation in `pandas/core/methods/to_dict.py` (lines 194-213) contains dead code that wastes computation. Specifically:

1. **Line 195-197**: The code computes an optimized `data` variable using `_create_data_for_split_and_tight_to_dict()` helper function
2. **Lines 203-209**: Instead of using the computed `data`, the code recomputes it inefficiently using `list(map(maybe_box_native, t))` for ALL values
3. **Performance Impact**: This bypasses the optimization in the helper function that only applies `maybe_box_native` to object-dtype columns when not all columns are object dtype

The helper function's docstring explicitly states it's for both 'split' and 'tight' orientations: "Simple helper method to create data for to `to_dict(orient='split')` and `to_dict(orient='tight')` to create the main output data" (lines 2008-2009 in `pandas/core/frame.py`).

The 'split' orientation correctly uses the computed `data` variable (line 190), achieving the intended performance optimization. The 'tight' orientation's failure to use this variable results in:
- Unnecessary computation of an unused variable (dead code)
- 3.3x slower performance on mixed-type DataFrames
- Inconsistent implementation between two orientations that should share the same data creation logic

## Relevant Context

The helper function `_create_data_for_split_and_tight_to_dict` in `pandas/core/frame.py` contains an important optimization (lines 2016-2024):
- When all columns are object dtype, it applies `maybe_box_native` during iteration
- When only some columns are object dtype, it first creates lists and then selectively applies `maybe_box_native` only to object columns
- This optimization avoids unnecessary boxing operations on numeric columns

The pandas documentation for `to_dict` doesn't specify any performance differences between 'split' and 'tight' orientations. The only documented difference is that 'tight' adds `index_names` and `column_names` metadata to the output dictionary.

Code references:
- Bug location: `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/methods/to_dict.py:195-209`
- Helper function: `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/frame.py:2004-2024`

## Proposed Fix

```diff
--- a/pandas/core/methods/to_dict.py
+++ b/pandas/core/methods/to_dict.py
@@ -199,13 +199,7 @@ def to_dict(
         return into_c(
             ((("index", df.index.tolist()),) if index else ())
             + (
                 ("columns", df.columns.tolist()),
-                (
-                    "data",
-                    [
-                        list(map(maybe_box_native, t))
-                        for t in df.itertuples(index=False, name=None)
-                    ],
-                ),
+                ("data", data),
             )
             + ((("index_names", list(df.index.names)),) if index else ())
             + (("column_names", list(df.columns.names)),)
```