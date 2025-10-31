# Bug Report: pandas.api.interchange UnicodeEncodeError with Surrogate Characters

**Target**: `pandas.api.interchange.from_dataframe`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The DataFrame interchange protocol crashes with a `UnicodeEncodeError` when attempting to convert DataFrames containing strings with UTF-16 surrogate characters (U+D800 to U+DFFF). While pandas DataFrames can store these strings, the interchange protocol fails to handle them, causing an unhandled exception.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st, settings
from pandas.api.interchange import from_dataframe

@given(
    st.data(),
    st.integers(min_value=2, max_value=5),
)
@settings(max_examples=50)
def test_roundtrip_unicode_strings(data, num_rows):
    unicode_strings = data.draw(st.lists(
        st.text(alphabet=st.characters(min_codepoint=0x0041, max_codepoint=0x1F600), min_size=0, max_size=20),
        min_size=num_rows,
        max_size=num_rows
    ))

    df = pd.DataFrame({'unicode_col': unicode_strings})

    interchange_obj = df.__dataframe__()
    result = from_dataframe(interchange_obj)

    pd.testing.assert_frame_equal(result, df)
```

**Failing input**: `['', '\ud800']`

## Reproducing the Bug

```python
import pandas as pd
from pandas.api.interchange import from_dataframe

df = pd.DataFrame({'str_col': ['hello', '\ud800']})
print(df)

interchange_obj = df.__dataframe__()
result = from_dataframe(interchange_obj)
```

**Output:**
```
UnicodeEncodeError: 'utf-8' codec can't encode character '\ud800' in position 0: surrogates not allowed
```

**Traceback:**
```
File "/pandas/core/interchange/from_dataframe.py", line 100, in from_dataframe
File "/pandas/core/interchange/from_dataframe.py", line 123, in _from_dataframe
File "/pandas/core/interchange/from_dataframe.py", line 175, in protocol_df_chunk_to_pandas
File "/pandas/core/interchange/from_dataframe.py", line 292, in string_column_to_ndarray
File "/pandas/core/interchange/column.py", line 287, in get_buffers
File "/pandas/core/interchange/column.py", line 351, in _get_data_buffer
    b.extend(obj.encode(encoding="utf-8"))
UnicodeEncodeError: 'utf-8' codec can't encode character '\ud800' in position 0: surrogates not allowed
```

## Why This Is A Bug

1. **Pandas accepts these strings**: You can create and manipulate DataFrames with surrogate characters without issues
2. **Unhandled exception**: The code crashes with `UnicodeEncodeError` instead of handling the error gracefully
3. **Inconsistent behavior**: Regular pandas operations work fine, but the interchange protocol fails
4. **No documentation**: There's no documented restriction against surrogate characters in the interchange protocol

The bug occurs in `/pandas/core/interchange/column.py` at line 351:
```python
b.extend(obj.encode(encoding="utf-8"))
```

This line assumes all strings are valid UTF-8, but UTF-16 surrogate characters are not valid UTF-8 and cause encoding to fail.

## Fix

Add error handling to gracefully manage strings that cannot be encoded to UTF-8. Several options:

**Option 1: Use surrogateescape error handler**
```diff
--- a/pandas/core/interchange/column.py
+++ b/pandas/core/interchange/column.py
@@ -348,7 +348,7 @@ class PandasColumn(Column):
             # TODO: this for-loop is slow; can be implemented in Cython/C/C++ later
             for obj in buf:
                 if isinstance(obj, str):
-                    b.extend(obj.encode(encoding="utf-8"))
+                    b.extend(obj.encode(encoding="utf-8", errors="surrogateescape"))
```

**Option 2: Raise a clear error with helpful message**
```diff
--- a/pandas/core/interchange/column.py
+++ b/pandas/core/interchange/column.py
@@ -348,7 +348,11 @@ class PandasColumn(Column):
             # TODO: this for-loop is slow; can be implemented in Cython/C/C++ later
             for obj in buf:
                 if isinstance(obj, str):
-                    b.extend(obj.encode(encoding="utf-8"))
+                    try:
+                        b.extend(obj.encode(encoding="utf-8"))
+                    except UnicodeEncodeError as e:
+                        raise ValueError(
+                            f"String column contains characters that cannot be encoded as UTF-8: {e}"
+                        ) from e
```

**Option 3: Use ignore error handler to skip invalid characters**
```diff
--- a/pandas/core/interchange/column.py
+++ b/pandas/core/interchange/column.py
@@ -348,7 +348,7 @@ class PandasColumn(Column):
             # TODO: this for-loop is slow; can be implemented in Cython/C/C++ later
             for obj in buf:
                 if isinstance(obj, str):
-                    b.extend(obj.encode(encoding="utf-8"))
+                    b.extend(obj.encode(encoding="utf-8", errors="ignore"))
```

Recommendation: **Option 1 (surrogateescape)** preserves data fidelity while handling the edge case, or **Option 2** if the interchange protocol spec explicitly requires valid UTF-8.