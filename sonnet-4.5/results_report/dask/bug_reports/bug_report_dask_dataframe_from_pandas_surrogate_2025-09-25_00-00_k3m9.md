# Bug Report: dask.dataframe.from_pandas Crashes on Surrogate Characters

**Target**: `dask.dataframe.from_pandas`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`dask.dataframe.from_pandas` crashes with a `UnicodeEncodeError` when converting pandas DataFrames containing string columns with Unicode surrogate characters (e.g., `\ud800`), despite these being valid pandas DataFrames.

## Property-Based Test

```python
from hypothesis import given, settings
from hypothesis.extra.pandas import column, data_frames, range_indexes
import dask.dataframe as dd


@given(
    data_frames([
        column('a', dtype=int),
        column('b', dtype=float),
        column('c', dtype=str)
    ], index=range_indexes(min_size=1, max_size=50))
)
@settings(max_examples=100, deadline=2000)
def test_from_pandas_roundtrip_dataframe(df):
    ddf = dd.from_pandas(df, npartitions=3)
    result = ddf.compute()
    pd.testing.assert_frame_equal(result, df, check_index_type=False)
```

**Failing input**: `pd.DataFrame({'a': [0], 'b': [0.0], 'c': ['\ud800']})`

## Reproducing the Bug

```python
import pandas as pd
import dask.dataframe as dd

df = pd.DataFrame({'text': ['\ud800']})
ddf = dd.from_pandas(df, npartitions=1)
```

**Output**:
```
UnicodeEncodeError: 'utf-8' codec can't encode character '\ud800' in position 0: surrogates not allowed
```

## Why This Is A Bug

1. **Valid pandas DataFrame**: The input is a valid pandas DataFrame that can be manipulated using standard pandas operations.

2. **Unexpected crash**: The `from_pandas` function should handle all valid pandas DataFrames, but crashes on this edge case without a helpful error message.

3. **Undocumented limitation**: The docstring for `from_pandas` does not mention any restrictions on string content or that PyArrow conversion might fail on certain valid Python strings.

4. **Real-world impact**: Surrogate characters can appear in real data from:
   - Corrupted UTF-8 data from web scraping
   - Binary data incorrectly interpreted as text
   - Legacy systems with non-standard encodings

## Fix

The bug occurs in `/dask/dataframe/_pyarrow.py` at line 69, where `df.astype(dtypes)` is called to convert object strings to PyArrow strings without validation. PyArrow's `pa.array()` function rejects surrogate characters as invalid UTF-8.

**Option 1: Graceful error handling**

```diff
--- a/dask/dataframe/_pyarrow.py
+++ b/dask/dataframe/_pyarrow.py
@@ -66,7 +66,12 @@ def _to_string_dtype(df, dtype_check, index_check, string_dtype):
             col: string_dtype for col, dtype in df.dtypes.items() if dtype_check(dtype)
         }
         if dtypes:
-            df = df.astype(dtypes)
+            try:
+                df = df.astype(dtypes)
+            except (UnicodeEncodeError, pa.lib.ArrowInvalid) as e:
+                raise ValueError(
+                    f"Cannot convert to PyArrow string dtype: {e}. "
+                    "Set dask.config.set({{'dataframe.convert-string': False}}) to disable PyArrow string conversion."
+                ) from e
     elif dtype_check(df.dtype):
         dtypes = string_dtype
         df = df.copy().astype(dtypes)
```

**Option 2: Silently skip invalid strings** (preserve original dtype when conversion fails)

```diff
--- a/dask/dataframe/_pyarrow.py
+++ b/dask/dataframe/_pyarrow.py
@@ -66,7 +66,11 @@ def _to_string_dtype(df, dtype_check, index_check, string_dtype):
             col: string_dtype for col, dtype in df.dtypes.items() if dtype_check(dtype)
         }
         if dtypes:
-            df = df.astype(dtypes)
+            try:
+                df = df.astype(dtypes)
+            except (UnicodeEncodeError, pa.lib.ArrowInvalid):
+                # Keep original dtype if PyArrow conversion fails
+                pass
     elif dtype_check(df.dtype):
         dtypes = string_dtype
         df = df.copy().astype(dtypes)
```