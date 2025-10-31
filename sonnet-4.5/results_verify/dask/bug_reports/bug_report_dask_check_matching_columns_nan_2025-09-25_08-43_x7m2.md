# Bug Report: dask.dataframe.utils.check_matching_columns Treats NaN Column Names as 0

**Target**: `dask.dataframe.utils.check_matching_columns`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`check_matching_columns` incorrectly treats column names containing `NaN` as equivalent to column names with value `0` due to the use of `np.nan_to_num`. This allows DataFrames with mismatched column names to pass validation silently.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
import pandas as pd
from dask.dataframe.utils import check_matching_columns

@given(st.lists(st.floats(allow_nan=False), min_size=1, max_size=5))
def test_check_matching_columns_nan_vs_zero(cols):
    meta_cols = [0] + cols
    actual_cols = [float('nan')] + cols

    meta = pd.DataFrame(columns=meta_cols)
    actual = pd.DataFrame(columns=actual_cols)

    try:
        check_matching_columns(meta, actual)
        assert False, f"Should raise ValueError for NaN vs 0 column mismatch"
    except ValueError:
        pass
```

**Failing input**: `meta.columns = [0, 1, 2]`, `actual.columns = [NaN, 1, 2]`

## Reproducing the Bug

```python
import numpy as np
import pandas as pd
from dask.dataframe.utils import check_matching_columns

meta = pd.DataFrame(columns=[0, 1, 2])
actual = pd.DataFrame(columns=[float('nan'), 1, 2])

check_matching_columns(meta, actual)
print("No error raised - BUG!")

print(f"meta.columns: {meta.columns.tolist()}")
print(f"actual.columns: {actual.columns.tolist()}")
print(f"Are they equal? {meta.columns.equals(actual.columns)}")
```

**Output:**
```
No error raised - BUG!
meta.columns: [0, 1, 2]
actual.columns: [nan, 1, 2]
Are they equal? False
```

The function does not raise a `ValueError` despite the columns being different.

## Why This Is A Bug

The bug occurs in line 395 of `dask/dataframe/utils.py`:

```python
if not np.array_equal(np.nan_to_num(meta.columns), np.nan_to_num(actual.columns)):
```

The `np.nan_to_num` function converts `NaN` values to `0` by default. This means:
- `meta.columns = [0, 1, 2]` → `nan_to_num` → `[0, 1, 2]`
- `actual.columns = [NaN, 1, 2]` → `nan_to_num` → `[0, 1, 2]`

After conversion, both arrays are `[0, 1, 2]`, so `np.array_equal` returns `True`, and the function returns without raising an error.

The comment on line 394 states:
```python
# Need nan_to_num otherwise nan comparison gives False
```

However, this comment conflates two distinct issues:
1. **NaN != NaN in floating point**: The IEEE 754 standard defines `NaN != NaN`
2. **Detecting mismatched columns**: We actually WANT to detect when one DataFrame has a `NaN` column and another has a different value

The current implementation incorrectly assumes that `NaN` column names should be treated as equivalent, when in fact they should be detected as mismatches.

## Fix

Use a NaN-aware comparison instead of `nan_to_num`. pandas Index objects have a proper equality check that handles NaN:

```diff
--- a/dask/dataframe/utils.py
+++ b/dask/dataframe/utils.py
@@ -391,8 +391,17 @@ def check_matching_columns(meta, actual):
 def check_matching_columns(meta, actual):
     import dask.dataframe.methods as methods

-    # Need nan_to_num otherwise nan comparison gives False
-    if not np.array_equal(np.nan_to_num(meta.columns), np.nan_to_num(actual.columns)):
+    # Use pandas Index.equals() which handles NaN correctly
+    # Two NaN values in the same position are considered equal,
+    # but NaN in one position and a different value in another are not
+    columns_match = (
+        len(meta.columns) == len(actual.columns) and
+        all(
+            (pd.isna(m) and pd.isna(a)) or m == a
+            for m, a in zip(meta.columns, actual.columns)
+        )
+    )
+    if not columns_match:
         extra = methods.tolist(actual.columns.difference(meta.columns))
         missing = methods.tolist(meta.columns.difference(actual.columns))
         if extra or missing:
```

Alternatively, use pandas' built-in Index equality which handles NaN properly:

```diff
--- a/dask/dataframe/utils.py
+++ b/dask/dataframe/utils.py
@@ -391,8 +391,8 @@ def check_matching_columns(meta, actual):
 def check_matching_columns(meta, actual):
     import dask.dataframe.methods as methods

-    # Need nan_to_num otherwise nan comparison gives False
-    if not np.array_equal(np.nan_to_num(meta.columns), np.nan_to_num(actual.columns)):
+    # pandas Index.equals handles NaN correctly (NaN==NaN in same position)
+    if not meta.columns.equals(actual.columns):
         extra = methods.tolist(actual.columns.difference(meta.columns))
         missing = methods.tolist(meta.columns.difference(actual.columns))
         if extra or missing:
```

The second fix is simpler and leverages pandas' existing NaN-aware equality checking.