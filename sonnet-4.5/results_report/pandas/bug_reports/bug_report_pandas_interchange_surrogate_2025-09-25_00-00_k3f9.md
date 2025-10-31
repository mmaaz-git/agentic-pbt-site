# Bug Report: pandas.api.interchange Surrogate Character Encoding

**Target**: `pandas.api.interchange.from_dataframe`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The interchange protocol crashes with a `UnicodeEncodeError` when attempting to convert pandas DataFrames containing surrogate characters (U+D800 through U+DFFF) back to pandas through `from_dataframe()`.

## Property-Based Test

```python
from hypothesis import given, settings
from hypothesis.extra.pandas import data_frames, column
import pandas as pd
import pandas.api.interchange as interchange


@given(data_frames([
    column('A', dtype=int),
    column('B', dtype=float),
    column('C', dtype=str),
]))
@settings(max_examples=200)
def test_from_dataframe_round_trip(df):
    """
    Property: from_dataframe(df.__dataframe__()) should equal df for pandas DataFrames
    Evidence: The docstring shows this is the intended usage pattern
    """
    interchange_obj = df.__dataframe__()
    result = interchange.from_dataframe(interchange_obj)
    pd.testing.assert_frame_equal(result, df)
```

**Failing input**: DataFrame with surrogate character `'\ud800'` in string column

## Reproducing the Bug

```python
import pandas as pd
import pandas.api.interchange as interchange

df = pd.DataFrame({'A': [0], 'B': [0.0], 'C': ['\ud800']})
interchange_obj = df.__dataframe__()
result = interchange.from_dataframe(interchange_obj)
```

Output:
```
UnicodeEncodeError: 'utf-8' codec can't encode character '\ud800' in position 0: surrogates not allowed
```

## Why This Is A Bug

1. **Valid pandas DataFrames are rejected**: Pandas DataFrames can legitimately contain surrogate characters in string columns, but the interchange protocol crashes when trying to convert them.

2. **Violates round-trip property**: The interchange protocol is designed to enable round-trip conversion between dataframe implementations. The documentation shows `from_dataframe(df.__dataframe__())` as the intended usage pattern, but this fails for DataFrames with surrogates.

3. **Poor error handling**: The code calls `str.encode('utf-8')` without handling potential `UnicodeEncodeError` exceptions, resulting in an unhelpful crash rather than a clear error message or graceful handling.

4. **Inconsistent with pandas design**: Pandas generally handles edge cases in string data gracefully, but the interchange protocol does not maintain this robustness.

## Fix

The issue is in `pandas/core/interchange/column.py` in the `PandasColumn._get_data_buffer()` method. The code currently uses:

```python
b.extend(obj.encode(encoding="utf-8"))
```

This should use an error handler to deal with surrogate characters:

```diff
--- a/pandas/core/interchange/column.py
+++ b/pandas/core/interchange/column.py
@@ -348,7 +348,7 @@
             # TODO: this for-loop is slow; can be implemented in Cython/C/C++ later
             for obj in buf:
                 if isinstance(obj, str):
-                    b.extend(obj.encode(encoding="utf-8"))
+                    b.extend(obj.encode(encoding="utf-8", errors="surrogatepass"))

             # Convert the byte array to a Pandas "buffer" using
             # a NumPy array as the backing store
```

The `errors="surrogatepass"` parameter allows surrogate characters to be encoded using the UTF-8 surrogate escape mechanism, which is the standard way to handle surrogates in UTF-8.