# Bug Report: pandas.from_dummies fails with empty DataFrame

**Target**: `pandas.core.reshape.encoding.from_dummies`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`from_dummies` fails to invert `get_dummies(..., drop_first=True)` when all categorical columns have only one unique value, violating the documented contract that it "inverts the operation performed by get_dummies".

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from hypothesis.extra.pandas import column, data_frames, range_indexes
import pandas as pd


@given(
    df=data_frames(
        columns=[
            column("A", elements=st.sampled_from(["cat", "dog", "bird"])),
            column("B", elements=st.sampled_from(["x", "y", "z"])),
        ],
        index=range_indexes(min_size=1, max_size=20),
    )
)
@settings(max_examples=200)
def test_get_dummies_from_dummies_with_drop_first(df):
    """
    Property: from_dummies should invert get_dummies(drop_first=True).
    Evidence: encoding.py line 376 states from_dummies "Inverts the operation
    performed by :func:`~pandas.get_dummies`."
    """
    dummies = pd.get_dummies(df, drop_first=True, dtype=int)

    default_cats = {}
    for col in df.columns:
        first_val = sorted(df[col].unique())[0]
        default_cats[col] = first_val

    recovered = pd.from_dummies(dummies, sep="_", default_category=default_cats)

    pd.testing.assert_frame_equal(recovered, df)
```

**Failing input**: `df = pd.DataFrame({"A": ["cat"], "B": ["x"]})`

## Reproducing the Bug

```python
import pandas as pd

df = pd.DataFrame({"A": ["cat"], "B": ["x"]})
dummies = pd.get_dummies(df, drop_first=True, dtype=int)

default_cats = {"A": "cat", "B": "x"}
recovered = pd.from_dummies(dummies, sep="_", default_category=default_cats)
```

**Error**:
```
ValueError: Length of 'default_category' (2) did not match the length of the columns being encoded (0)
```

## Why This Is A Bug

1. **Contract Violation**: The docstring at line 376 in `encoding.py` explicitly states that `from_dummies` "Inverts the operation performed by :func:`~pandas.get_dummies`".

2. **Expected Behavior**: When `get_dummies(..., drop_first=True)` is called on data where each column has only one unique value, it returns an empty DataFrame (see `encoding.py` line 295-296). The inverse operation `from_dummies` should be able to reconstruct the original data using only the `default_category` parameter.

3. **Actual Behavior**: `from_dummies` raises a `ValueError` because it checks that `len(default_category) == len(variables_slice)` (line 518), where `variables_slice` is empty when the input DataFrame has no columns.

4. **Real-World Impact**: This can occur when a user filters a dataset to a subset that happens to have only one unique value per categorical column, then tries to use the standard `get_dummies`/`from_dummies` round-trip pattern.

## Fix

The bug is in the length check at line 518 of `encoding.py`. When the input DataFrame is empty (has no columns), `from_dummies` should still be able to reconstruct the data using the `default_category` dict.

```diff
--- a/pandas/core/reshape/encoding.py
+++ b/pandas/core/reshape/encoding.py
@@ -515,12 +515,17 @@ def from_dummies(

     if default_category is not None:
         if isinstance(default_category, dict):
-            if not len(default_category) == len(variables_slice):
+            # When data is empty (all columns dropped), variables_slice is empty
+            # but we should still accept default_category to reconstruct the data
+            if len(variables_slice) > 0 and not len(default_category) == len(variables_slice):
                 len_msg = (
                     f"Length of 'default_category' ({len(default_category)}) "
                     f"did not match the length of the columns being encoded "
                     f"({len(variables_slice)})"
                 )
                 raise ValueError(len_msg)
+            elif len(variables_slice) == 0:
+                # Reconstruct using only default_category
+                return pd.DataFrame({k: [v] * len(data) for k, v in default_category.items()}, index=data.index)
         elif isinstance(default_category, Hashable):
             default_category = dict(
                 zip(variables_slice, [default_category] * len(variables_slice))
```