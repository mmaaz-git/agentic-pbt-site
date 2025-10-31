# Bug Report: dask.dataframe.from_pandas Unicode Surrogate Crash

**Target**: `dask.dataframe.from_pandas`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`dask.dataframe.from_pandas` crashes with a UnicodeEncodeError when the input pandas DataFrame contains Unicode surrogate characters, even though pandas itself supports storing such characters.

## Property-Based Test

```python
import dask.dataframe as dd
import pandas as pd
from hypothesis import given, strategies as st
from hypothesis.extra.pandas import data_frames, column, range_indexes

@given(
    data_frames([
        column('a', dtype=int),
        column('b', dtype=float),
        column('c', dtype=str),
    ], index=range_indexes(min_size=1, max_size=100))
)
def test_from_pandas_compute_roundtrip(pdf):
    ddf = dd.from_pandas(pdf, npartitions=2)
    result = ddf.compute()
    pd.testing.assert_frame_equal(result, pdf)
```

**Failing input**: DataFrame with a string column containing `'\ud800'` (a Unicode surrogate character)

## Reproducing the Bug

```python
import dask.dataframe as dd
import pandas as pd

pdf = pd.DataFrame({'a': [0], 'b': [0.0], 'c': ['\ud800']})

ddf = dd.from_pandas(pdf, npartitions=2)
```

Expected: Successfully creates a Dask DataFrame
Actual: Raises `UnicodeEncodeError: 'utf-8' codec can't encode character '\ud800' in position 0: surrogates not allowed`

## Why This Is A Bug

Pandas DataFrames can store arbitrary Python strings, including those with Unicode surrogate characters. The following code works correctly:

```python
import pandas as pd
pdf = pd.DataFrame({'c': ['\ud800']})
assert pdf['c'].iloc[0] == '\ud800'
```

Since `from_pandas` accepts pandas DataFrames as input, it should handle all valid pandas DataFrames without crashing. The crash occurs during internal dtype conversion to PyArrow strings, which do not support surrogates.

## Fix

The issue occurs in `_to_string_dtype` in `dask/dataframe/_pyarrow.py` when converting object dtypes to PyArrow string dtypes. Dask should either:

1. Catch the UnicodeEncodeError and fall back to keeping the original object dtype
2. Document that string columns with surrogate characters are not supported
3. Validate input and raise a clear error message before processing

Option 1 (graceful fallback) would be the most user-friendly:

```diff
--- a/dask/dataframe/_pyarrow.py
+++ b/dask/dataframe/_pyarrow.py
@@ -66,7 +66,11 @@ def _to_string_dtype(df):
     dtypes = {
         col: pd.StringDtype("pyarrow") for col, dt in df.dtypes.items() if dt == object
     }
-    df = df.astype(dtypes)
+    try:
+        df = df.astype(dtypes)
+    except (UnicodeEncodeError, UnicodeDecodeError):
+        # Fallback to object dtype if conversion fails
+        pass
     return df
```