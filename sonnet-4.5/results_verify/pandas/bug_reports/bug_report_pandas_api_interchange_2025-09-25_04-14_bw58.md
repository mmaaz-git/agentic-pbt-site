# Bug Report: pandas.api.interchange UnicodeEncodeError with Surrogate Characters

**Target**: `pandas.api.interchange.from_dataframe`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The pandas interchange protocol crashes with a `UnicodeEncodeError` when attempting to convert DataFrames containing string columns with Unicode surrogate characters (U+D800 to U+DFFF). This prevents round-trip conversion of valid pandas DataFrames through the interchange protocol.

## Property-Based Test

```python
import pandas as pd
import pandas.api.interchange as interchange
from hypothesis import given, settings
import hypothesis.extra.pandas as pdst


@settings(max_examples=500)
@given(pdst.data_frames(
    columns=[
        pdst.column('A', dtype=int),
        pdst.column('B', dtype=float),
        pdst.column('C', dtype=str),
    ]
))
def test_round_trip_from_dataframe(df):
    interchange_obj = df.__dataframe__()
    result = interchange.from_dataframe(interchange_obj)

    assert result.shape == df.shape, "Shape should be preserved"
    assert list(result.columns) == list(df.columns), "Column names should be preserved"

    pd.testing.assert_frame_equal(result, df, check_dtype=True)
```

**Failing input**: DataFrame with a string column containing `'\ud800'` (a surrogate character)

## Reproducing the Bug

```python
import pandas as pd
import pandas.api.interchange as interchange

df = pd.DataFrame({
    'A': [0],
    'B': [0.0],
    'C': ['\ud800']
})

interchange_obj = df.__dataframe__()
result = interchange.from_dataframe(interchange_obj)
```

Expected: The DataFrame should be successfully converted through the interchange protocol.

Actual:
```
UnicodeEncodeError: 'utf-8' codec can't encode character '\ud800' in position 0: surrogates not allowed
```

## Why This Is A Bug

1. **Pandas allows surrogate characters**: DataFrame creation with surrogate characters succeeds, so they are valid data.

2. **Round-trip should preserve data**: The docstring example for `from_dataframe` demonstrates round-trip usage: `from_dataframe(df.__dataframe__())` should preserve the original DataFrame.

3. **Violates interchange protocol contract**: The interchange protocol is meant to transfer DataFrames between different libraries, but it cannot handle all valid pandas DataFrames.

The root cause is in `/pandas/core/interchange/column.py` at line ~351 in the `_get_data_buffer` method:

```python
for obj in buf:
    if isinstance(obj, str):
        b.extend(obj.encode(encoding="utf-8"))
```

The `encode("utf-8")` call fails for surrogate characters because UTF-8 encoding does not permit lone surrogates.

## Fix

```diff
--- a/pandas/core/interchange/column.py
+++ b/pandas/core/interchange/column.py
@@ -348,7 +348,7 @@ class PandasColumn:
             # TODO: this for-loop is slow; can be implemented in Cython/C/C++ later
             for obj in buf:
                 if isinstance(obj, str):
-                    b.extend(obj.encode(encoding="utf-8"))
+                    b.extend(obj.encode(encoding="utf-8", errors="surrogatepass"))

             # Convert the byte array to a Pandas "buffer" using
             # a NumPy array as the backing store
```

The `surrogatepass` error handler allows encoding of surrogate characters by preserving them in the UTF-8 output, enabling proper round-trip conversion.