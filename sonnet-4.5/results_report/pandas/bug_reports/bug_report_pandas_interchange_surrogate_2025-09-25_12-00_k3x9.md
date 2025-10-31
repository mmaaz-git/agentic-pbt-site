# Bug Report: pandas.core.interchange Crashes on Strings with Lone Surrogates

**Target**: `pandas.core.interchange.column.PandasColumn._get_data_buffer`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The pandas interchange protocol implementation crashes with a `UnicodeEncodeError` when attempting to convert DataFrames containing strings with lone surrogate characters (e.g., `\ud800`). While such strings are uncommon, they are legal in Python and can occur when reading files with broken encodings.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd
import pytest


@given(st.text(alphabet=st.sampled_from(['\ud800', '\udc00', '\udfff']), min_size=1, max_size=10))
def test_string_with_surrogates_crashes(surrogate_str):
    df = pd.DataFrame({"str_col": [surrogate_str]})
    interchange_obj = df.__dataframe__()

    with pytest.raises(UnicodeEncodeError):
        pd.api.interchange.from_dataframe(interchange_obj)
```

**Failing input**: `"\ud800"` (a lone high surrogate character)

## Reproducing the Bug

```python
import pandas as pd

df = pd.DataFrame({"str_col": ["\ud800"]})
interchange_obj = df.__dataframe__()
result = pd.api.interchange.from_dataframe(interchange_obj)
```

This raises:
```
UnicodeEncodeError: 'utf-8' codec can't encode character '\ud800' in position 0: surrogates not allowed
```

The error occurs at `pandas/core/interchange/column.py:351` in the `_get_data_buffer` method.

## Why This Is A Bug

1. **Poor error handling**: The error occurs deep in the implementation without a clear message about the actual issue
2. **Inconsistent behavior**: Pandas allows storing such strings but the interchange protocol cannot handle them
3. **Better alternatives exist**: The code could either:
   - Validate strings early and provide a clear error message like "Interchange protocol requires valid UTF-8; string contains lone surrogate at position X"
   - Use `errors='surrogatepass'` in the encode call to handle surrogates gracefully
   - Document this limitation clearly in the function docstring

## Fix

Option 1: Handle surrogates gracefully using `surrogatepass`:

```diff
--- a/pandas/core/interchange/column.py
+++ b/pandas/core/interchange/column.py
@@ -348,7 +348,7 @@ class PandasColumn(Column):
             # TODO: this for-loop is slow; can be implemented in Cython/C/C++ later
             for obj in buf:
                 if isinstance(obj, str):
-                    b.extend(obj.encode(encoding="utf-8"))
+                    b.extend(obj.encode(encoding="utf-8", errors="surrogatepass"))

             # Convert the byte array to a Pandas "buffer" using
             # a NumPy array as the backing store
@@ -436,7 +436,7 @@ class PandasColumn(Column):
                 # For missing values (in this case, `np.nan` values)
                 # we don't increment the pointer
                 if isinstance(v, str):
-                    b = v.encode(encoding="utf-8")
+                    b = v.encode(encoding="utf-8", errors="surrogatepass")
                     ptr += len(b)

                 offsets[i + 1] = ptr
```

Option 2: Validate early and provide a clear error message:

```diff
--- a/pandas/core/interchange/column.py
+++ b/pandas/core/interchange/column.py
@@ -343,10 +343,18 @@ class PandasColumn(Column):
         elif self.dtype[0] == DtypeKind.STRING:
             # Marshal the strings from a NumPy object array into a byte array
             buf = self._col.to_numpy()
             b = bytearray()

             # TODO: this for-loop is slow; can be implemented in Cython/C/C++ later
             for obj in buf:
                 if isinstance(obj, str):
-                    b.extend(obj.encode(encoding="utf-8"))
+                    try:
+                        b.extend(obj.encode(encoding="utf-8"))
+                    except UnicodeEncodeError as e:
+                        raise ValueError(
+                            f"Interchange protocol requires valid UTF-8 strings. "
+                            f"String contains invalid UTF-8 sequence: {e}"
+                        ) from e

             # Convert the byte array to a Pandas "buffer" using
             # a NumPy array as the backing store
```

The first option (using `surrogatepass`) is preferable as it handles the data gracefully while maintaining round-trip capability. The interchange protocol counterpart (`from_dataframe.py:349`) already decodes using UTF-8, which will handle surrogate-passed data correctly.