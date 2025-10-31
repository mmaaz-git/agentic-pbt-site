# Bug Report: pandas.core.interchange Unicode Surrogate Encoding Crash

**Target**: `pandas.core.interchange.column.PandasColumn._get_data_buffer`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The pandas interchange protocol crashes with a `UnicodeEncodeError` when attempting to encode strings containing Unicode surrogate characters (U+D800-U+DFFF) during DataFrame interchange conversion.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st, settings
from pandas.core.interchange.from_dataframe import from_dataframe

@given(st.data())
@settings(max_examples=200)
def test_from_dataframe_with_unicode_strings(data):
    n_rows = data.draw(st.integers(min_value=1, max_value=20))

    values = []
    for _ in range(n_rows):
        val = data.draw(st.text(
            alphabet=st.characters(min_codepoint=0x0000, max_codepoint=0x1FFFF),
            min_size=0,
            max_size=20
        ))
        values.append(val)

    df_original = pd.DataFrame({'col': values})
    interchange_obj = df_original.__dataframe__()
    df_roundtrip = from_dataframe(interchange_obj)

    pd.testing.assert_frame_equal(df_original, df_roundtrip)
```

**Failing input**: `'\ud800'` (and other surrogate characters in range U+D800-U+DFFF)

## Reproducing the Bug

```python
import pandas as pd
from pandas.core.interchange.from_dataframe import from_dataframe

df = pd.DataFrame({'col': ['\ud800']})
interchange_obj = df.__dataframe__()
df_roundtrip = from_dataframe(interchange_obj)
```

**Output:**
```
UnicodeEncodeError: 'utf-8' codec can't encode character '\ud800' in position 0: surrogates not allowed
```

## Why This Is A Bug

1. Pandas accepts and stores strings containing surrogate characters without issue
2. The interchange protocol should preserve data during round-trip conversion: `from_dataframe(df.__dataframe__()) == df`
3. The crash occurs in `pandas/core/interchange/column.py:351` when encoding strings to UTF-8
4. Users who have DataFrames with such strings (even if uncommon) cannot use the interchange protocol

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

             # Convert the byte array to a NumPy array
             data = np.frombuffer(b, dtype=np.uint8)
```

The `surrogatepass` error handler allows encoding and decoding of surrogate characters, which is appropriate for the interchange protocol that needs to faithfully preserve data even when it contains technically invalid Unicode sequences.