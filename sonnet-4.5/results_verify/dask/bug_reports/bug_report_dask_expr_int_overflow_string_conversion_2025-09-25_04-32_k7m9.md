# Bug Report: dask.dataframe.dask_expr Integer Overflow to String Conversion

**Target**: `dask.dataframe.dask_expr.from_pandas`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When a pandas Series contains integers that overflow int64 (e.g., `-9223372036854775809`), `from_pandas` incorrectly converts them to PyArrow strings instead of preserving the object dtype. This results in data corruption where integer values become strings.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd
import dask.dataframe.dask_expr as dex

@settings(max_examples=100)
@given(
    st.lists(st.integers(), min_size=1, max_size=50),
    st.integers(min_value=1, max_value=5)
)
def test_from_pandas_round_trip_series(data, npartitions):
    s = pd.Series(data)
    dask_s = dex.from_pandas(s, npartitions=npartitions, sort=False)
    result = dask_s.compute()
    pd.testing.assert_series_equal(result, s, check_index_type=False)
```

**Failing input**: `data=[-9223372036854775809], npartitions=1`

## Reproducing the Bug

```python
import pandas as pd
import dask.dataframe.dask_expr as dex

s = pd.Series([-9223372036854775809])
print(f"Original dtype: {s.dtype}")
print(f"Original value: {s.values[0]} (type: {type(s.values[0])})")

dask_s = dex.from_pandas(s, npartitions=1, sort=False)
result = dask_s.compute()

print(f"Result dtype: {result.dtype}")
print(f"Result value: {result.values[0]} (type: {type(result.values[0])})")

assert s.dtype == result.dtype
```

## Why This Is A Bug

1. **Data Corruption**: Integer values are converted to strings, changing their semantic meaning
2. **Type Inconsistency**: The original Series has `object` dtype (containing integers), but the result has `StringDtype(storage=pyarrow)`
3. **Round-trip Failure**: `from_pandas(s).compute()` should return data equivalent to the original `s`, but instead returns a completely different data type
4. **Silent Failure**: No warning or error is raised, making this bug particularly dangerous

The root cause is in `/dask/dataframe/_pyarrow.py`:
- The `is_object_string_dtype` function incorrectly identifies object dtypes containing large integers as string dtypes
- This causes `to_pyarrow_string` to convert the data to PyArrow strings
- The conversion happens when `pyarrow_strings_enabled()` returns `True` (the default)

## Fix

The fix should modify `is_object_string_dtype` in `_pyarrow.py` to check if the object dtype actually contains string data, not just rely on `pd.api.types.is_string_dtype` which can give false positives for object dtypes.

One approach would be to inspect the actual data when dtype is `object` to determine if it's truly string data:

```diff
--- a/dask/dataframe/_pyarrow.py
+++ b/dask/dataframe/_pyarrow.py
@@ -19,10 +19,16 @@ def is_pyarrow_string_dtype(dtype):

 def is_object_string_dtype(dtype):
     """Determine if input is a non-pyarrow string dtype"""
+    # Don't convert object dtypes - they may contain non-string data
+    # like large integers that overflow int64
+    if dtype == 'object':
+        return False
+
     # in pandas < 2.0, is_string_dtype(DecimalDtype()) returns True
     return (
         pd.api.types.is_string_dtype(dtype)
         and not is_pyarrow_string_dtype(dtype)
         and not pd.api.types.is_dtype_equal(dtype, "decimal")
     )
```

A more sophisticated fix would inspect sample values to determine if the object dtype truly contains strings.