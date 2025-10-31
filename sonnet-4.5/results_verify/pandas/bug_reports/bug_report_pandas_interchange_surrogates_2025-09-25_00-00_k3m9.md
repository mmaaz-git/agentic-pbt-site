# Bug Report: pandas.api.interchange String Encoding Crash with Surrogate Characters

**Target**: `pandas.api.interchange.from_dataframe`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The interchange protocol crashes with a `UnicodeEncodeError` when attempting to convert DataFrames containing string columns with Unicode surrogate characters (U+D800 to U+DFFF), which are valid Python strings but cannot be encoded to UTF-8.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from hypothesis.extra.pandas import data_frames, column
import pandas as pd

@given(data_frames(columns=[
    column('int_col', dtype=int),
    column('float_col', dtype=float),
    column('str_col', dtype=str)
]))
def test_round_trip_mixed_types(df):
    """Round-trip should preserve mixed column types."""
    interchange_obj = df.__dataframe__()
    result = pd.api.interchange.from_dataframe(interchange_obj)

    assert result.shape == df.shape
    assert list(result.columns) == list(df.columns)
```

**Failing input**: A DataFrame with string column containing `'\ud800'` (or any surrogate character)

## Reproducing the Bug

```python
import pandas as pd

df = pd.DataFrame({
    'int_col': [0],
    'float_col': [0.0],
    'str_col': ['\ud800']
})

interchange_obj = df.__dataframe__()
result = pd.api.interchange.from_dataframe(interchange_obj)
```

**Output**:
```
UnicodeEncodeError: 'utf-8' codec can't encode character '\ud800' in position 0: surrogates not allowed
```

## Why This Is A Bug

1. **Valid Python strings**: Surrogate characters are valid in Python strings (Python internally uses a flexible string representation that can contain surrogates)
2. **Pandas accepts the data**: Pandas DataFrames can store these strings without issue
3. **Unhandled crash**: The interchange protocol crashes instead of handling the case gracefully or providing a clear error message
4. **Real-world impact**: Data from external sources (e.g., databases, files with encoding issues) can contain surrogates

## Fix

The bug occurs in `pandas/core/interchange/column.py` at line 351 in the `_get_data_buffer` method:

```python
b.extend(obj.encode(encoding="utf-8"))
```

**Option 1: Handle surrogates using surrogatepass**
```diff
--- a/pandas/core/interchange/column.py
+++ b/pandas/core/interchange/column.py
@@ -348,7 +348,7 @@ class PandasColumn:
         for i, obj in enumerate(column):
             if isinstance(obj, str):
                 offset += ptr.itemsize
-                b.extend(obj.encode(encoding="utf-8"))
+                b.extend(obj.encode(encoding="utf-8", errors="surrogatepass"))
                 ptr[i + 1] = offset
             else:
                 offset += ptr.itemsize
```

**Option 2: Use replace to handle invalid characters**
```diff
--- a/pandas/core/interchange/column.py
+++ b/pandas/core/interchange/column.py
@@ -348,7 +348,7 @@ class PandasColumn:
         for i, obj in enumerate(column):
             if isinstance(obj, str):
                 offset += ptr.itemsize
-                b.extend(obj.encode(encoding="utf-8"))
+                b.extend(obj.encode(encoding="utf-8", errors="replace"))
                 ptr[i + 1] = offset
             else:
                 offset += ptr.itemsize
```

**Option 3: Raise a clear error early**
```diff
--- a/pandas/core/interchange/column.py
+++ b/pandas/core/interchange/column.py
@@ -348,7 +348,11 @@ class PandasColumn:
         for i, obj in enumerate(column):
             if isinstance(obj, str):
                 offset += ptr.itemsize
-                b.extend(obj.encode(encoding="utf-8"))
+                try:
+                    b.extend(obj.encode(encoding="utf-8"))
+                except UnicodeEncodeError as e:
+                    raise ValueError(f"String contains characters that cannot be encoded to UTF-8: {e}. "
+                                   "The interchange protocol requires valid UTF-8 strings.") from e
                 ptr[i + 1] = offset
             else:
                 offset += ptr.itemsize
```

I recommend **Option 1** (surrogatepass) as it preserves data fidelity while allowing the interchange to work, though the receiving system must also handle surrogates correctly.