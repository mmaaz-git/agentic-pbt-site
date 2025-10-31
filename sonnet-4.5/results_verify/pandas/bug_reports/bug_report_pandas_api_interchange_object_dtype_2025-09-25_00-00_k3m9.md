# Bug Report: pandas.api.interchange Object Dtype Validation

**Target**: `pandas.api.interchange.from_dataframe`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`from_dataframe()` fails with `NotImplementedError` when given an interchange object created from a DataFrame with object dtype containing non-string values (e.g., large integers). The error occurs late in the conversion process rather than when the interchange object is created, leading to poor user experience.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
import pandas as pd
import pandas.api.interchange as interchange

@given(st.lists(st.integers(), min_size=1, max_size=100))
@settings(max_examples=100)
def test_single_column_roundtrip(values):
    df = pd.DataFrame({'col': values})
    result = interchange.from_dataframe(df.__dataframe__())
    assert result.shape == df.shape
```

**Failing input**: `values=[-9_223_372_036_854_775_809]`

## Reproducing the Bug

```python
import pandas as pd
import pandas.api.interchange as interchange

df = pd.DataFrame({'col': [-9_223_372_036_854_775_809]})

interchange_obj = df.__dataframe__()

result = interchange.from_dataframe(interchange_obj)
```

**Output:**
```
NotImplementedError: Non-string object dtypes are not supported yet
```

## Why This Is A Bug

This violates expected behavior in several ways:

1. **Late error detection**: `df.__dataframe__()` succeeds, but `from_dataframe()` fails. Users would expect the error to be raised when creating the interchange object if the dtype is unsupported.

2. **Inconsistent round-trip behavior**: For a pandas DataFrame with object dtype containing integers, users might reasonably expect the interchange protocol to either:
   - Reject it early (fail fast), or
   - Handle it gracefully (e.g., by detecting that the objects are integers and converting appropriately)

3. **Poor error message**: The message "Non-string object dtypes are not supported yet" doesn't help users understand how to fix the problem or what to do instead.

4. **Arbitrary Python integers are valid in pandas**: When creating a DataFrame with Python integers that exceed int64 range, pandas accepts them and stores them in object dtype. The interchange protocol should either validate this earlier or handle it better.

## Why This Matters

- **Confusing failure mode**: The interchange object is created successfully but cannot be converted back
- **Large integers in data**: Users with database IDs, timestamps, or other large numbers might encounter this unexpectedly
- **Inconsistent with design principles**: The interchange protocol should either fail fast or handle edge cases gracefully

## Fix

Add early validation in the DataFrame interchange implementation to check if all columns have supported dtypes before creating the interchange object. This would provide better error messages and fail-fast behavior.

Suggested change in `pandas/core/interchange/dataframe.py` (or wherever `__dataframe__` is implemented):

```diff
def __dataframe__(self, allow_copy: bool = True):
+   # Validate that all dtypes are supported by the interchange protocol
+   for col_name in self.column_names():
+       col = self.get_column_by_name(col_name)
+       dtype = self._col.dtype
+       if dtype == object:
+           # Check if this is a string dtype (supported) or other object dtype (not supported)
+           from pandas.api.types import is_string_dtype
+           from pandas.core.dtypes.inference import infer_dtype
+           if infer_dtype(self._col) not in ("string", "empty"):
+               raise TypeError(
+                   f"Column '{col_name}' has dtype '{dtype}' which is not supported "
+                   "by the interchange protocol. Consider converting to a supported dtype "
+                   "(int64, float64, bool, string, categorical, or datetime64)."
+               )
    return PandasDataFrame(self, allow_copy=allow_copy)
```

Alternatively, consider supporting object dtype with integers by detecting them and converting to an appropriate integer type (e.g., int64 if they fit, or raising a more specific error if they don't).