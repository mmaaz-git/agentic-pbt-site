# Bug Report: pandas.core.methods.to_dict 'tight' Orient Unused Variable

**Target**: `pandas.core.methods.to_dict`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `to_dict` function's 'tight' orient implementation computes a `data` variable using an optimized helper method but never uses it, instead recomputing the data less efficiently. This wastes computation and bypasses important performance optimizations.

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
    split_result = df.to_dict(orient='split')
    tight_result = df.to_dict(orient='tight')

    assert split_result['data'] == tight_result['data']
```

**Failing input**: N/A - this is a code quality/performance bug, not a correctness bug

## Reproducing the Bug

The bug can be seen by examining the source code of `to_dict` in `pandas/core/methods/to_dict.py`:

Lines 194-213 ('tight' orient):
```python
elif orient == "tight":
    data = df._create_data_for_split_and_tight_to_dict(
        are_all_object_dtype_cols, box_native_indices
    )

    return into_c(
        ((("index", df.index.tolist()),) if index else ())
        + (
            ("columns", df.columns.tolist()),
            (
                "data",
                [
                    list(map(maybe_box_native, t))
                    for t in df.itertuples(index=False, name=None)
                ],
            ),
        )
        + ((("index_names", list(df.index.names)),) if index else ())
        + (("column_names", list(df.columns.names)),)
    )
```

Compare to lines 181-192 ('split' orient):
```python
elif orient == "split":
    data = df._create_data_for_split_and_tight_to_dict(
        are_all_object_dtype_cols, box_native_indices
    )

    return into_c(
        ((("index", df.index.tolist()),) if index else ())
        + (
            ("columns", df.columns.tolist()),
            ("data", data),
        )
    )
```

## Why This Is A Bug

1. **Wasted Computation**: The `data` variable is computed on line 195 but never used
2. **Bypassed Optimization**: The helper method `_create_data_for_split_and_tight_to_dict` has an optimization where it only applies `maybe_box_native` to columns with object dtype when `are_all_object_dtype_cols=False`. But the 'tight' orient always applies `maybe_box_native` to ALL columns (line 206), bypassing this optimization
3. **Inconsistency**: The 'split' orient correctly uses the computed `data` variable, but 'tight' doesn't

The helper method optimization (in `pandas/core/frame.py` lines 2016-2023):
```python
else:
    data = [list(t) for t in self.itertuples(index=False, name=None)]
    if object_dtype_indices:
        # If we have object_dtype_cols, apply maybe_box_naive after list
        # comprehension for perf
        for row in data:
            for i in object_dtype_indices:
                row[i] = maybe_box_native(row[i])
```

This optimization is lost when 'tight' recomputes the data.

## Fix

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