# Bug Report: pandas.api.interchange Boolean Null Values Silently Converted to False

**Target**: `pandas.core.interchange.from_dataframe.set_nulls`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Null values in nullable boolean columns are silently converted to `False` instead of being preserved as null/NA when using the DataFrame interchange protocol. This happens because NumPy silently converts `None` to `False` in boolean arrays instead of raising a TypeError.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd
from pandas.api.interchange import from_dataframe


@given(st.lists(st.one_of(st.booleans(), st.none()), min_size=1, max_size=100))
def test_round_trip_nullable_bool(bool_list):
    df = pd.DataFrame({"col": pd.array(bool_list, dtype="boolean")})
    result = from_dataframe(df.__dataframe__())
    pd.testing.assert_frame_equal(result, df)
```

**Failing input**: `bool_list=[True, False, None]`

## Reproducing the Bug

```python
import pandas as pd
from pandas.api.interchange import from_dataframe

df = pd.DataFrame({"col": pd.array([True, False, None], dtype="boolean")})
print("Original:", df["col"].tolist())

result = from_dataframe(df.__dataframe__())
print("After round-trip:", result["col"].tolist())
```

**Output:**
```
Original: [True, False, <NA>]
After round-trip: [True, False, False]
```

## Why This Is A Bug

The DataFrame interchange protocol should preserve null values through round-trip conversions. The `set_nulls` function attempts to set null values by assigning `None` to array positions, expecting a TypeError for non-nullable dtypes:

```python
try:
    data[null_pos] = None
except TypeError:
    data = data.astype(float)
    data[null_pos] = None
```

However, NumPy's boolean type silently converts `None` to `False` instead of raising TypeError:

```python
>>> import numpy as np
>>> arr = np.array([True, False, False])
>>> arr[[False, False, True]] = None
>>> arr
array([ True, False, False])  # None was converted to False!
```

This causes null values to be lost without warning.

## Fix

The fix is to check for boolean dtype explicitly and convert to a nullable representation before setting nulls:

```diff
--- a/pandas/core/interchange/from_dataframe.py
+++ b/pandas/core/interchange/from_dataframe.py
@@ -540,6 +540,12 @@ def set_nulls(

     if null_pos is not None and np.any(null_pos):
         if not allow_modify_inplace:
             data = data.copy()
+
+        # NumPy bool dtype silently converts None to False instead of raising TypeError
+        # so we need to handle it explicitly
+        if isinstance(data, np.ndarray) and data.dtype == np.bool_:
+            data = data.astype(float)
+
         try:
             data[null_pos] = None
         except TypeError:
```