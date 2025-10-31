# Bug Report: pandas.api.interchange UnicodeEncodeError on Surrogate Characters

**Target**: `pandas.api.interchange.from_dataframe`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `from_dataframe` function crashes with `UnicodeEncodeError` when processing DataFrames containing string columns with UTF-16 surrogate characters (U+D800 to U+DFFF), violating the round-trip conversion property.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd
from pandas.api.interchange import from_dataframe
from hypothesis import given, strategies as st, settings
from hypothesis.extra.pandas import data_frames, column


@given(data_frames([
    column('a', dtype=int),
    column('b', dtype=float),
    column('c', dtype=str),
]))
@settings(max_examples=100)
def test_column_names_preserved(df):
    interchange_obj = df.__dataframe__()
    result = from_dataframe(interchange_obj)

    assert list(interchange_obj.column_names()) == list(result.columns)
```

**Failing input**: DataFrame with a string column containing `'\ud800'` (surrogate character)

## Reproducing the Bug

```python
import pandas as pd
from pandas.api.interchange import from_dataframe

df = pd.DataFrame({'col': ['\ud800']})
interchange_obj = df.__dataframe__()
result = from_dataframe(interchange_obj)
```

## Why This Is A Bug

UTF-16 surrogate characters (U+D800 to U+DFFF) are valid Python string characters and can be stored in pandas DataFrames. However, the interchange protocol's string encoding implementation attempts to encode these characters as UTF-8, which fails because surrogates cannot be represented in UTF-8.

This violates the expected round-trip property: `from_dataframe(df.__dataframe__())` should equal `df` for any pandas DataFrame that pandas itself accepts.

The error occurs in `pandas/core/interchange/column.py` at line 351:

```python
b.extend(obj.encode(encoding="utf-8"))
```

## Fix

The fix should handle surrogate characters gracefully, either by:
1. Using `errors='surrogatepass'` or `errors='replace'` when encoding
2. Validating and rejecting DataFrames with surrogate characters upfront with a clear error message
3. Using a different encoding that supports surrogates

Option 1 is the least invasive:

```diff
--- a/pandas/core/interchange/column.py
+++ b/pandas/core/interchange/column.py
@@ -348,7 +348,7 @@ class PandasColumn:
         # TODO: this for-loop is slow; can be implemented in Cython/C/C++ later
         for obj in buf:
             if isinstance(obj, str):
-                b.extend(obj.encode(encoding="utf-8"))
+                b.extend(obj.encode(encoding="utf-8", errors="surrogatepass"))
```

However, note that `surrogatepass` is only available on certain platforms. A more robust solution might be:

```diff
--- a/pandas/core/interchange/column.py
+++ b/pandas/core/interchange/column.py
@@ -348,7 +348,10 @@ class PandasColumn:
         # TODO: this for-loop is slow; can be implemented in Cython/C/C++ later
         for obj in buf:
             if isinstance(obj, str):
-                b.extend(obj.encode(encoding="utf-8"))
+                try:
+                    b.extend(obj.encode(encoding="utf-8"))
+                except UnicodeEncodeError:
+                    b.extend(obj.encode(encoding="utf-8", errors="surrogatepass"))
```