# Bug Report: pandas.api.interchange Surrogate Character Encoding

**Target**: `pandas.api.interchange.from_dataframe`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The pandas interchange protocol crashes with `UnicodeEncodeError` when attempting to convert a DataFrame containing surrogate characters in string columns back from the interchange format.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from hypothesis.extra.pandas import data_frames, columns, column
import pandas.api.interchange as interchange

@given(data_frames([
    column('A', dtype=int),
    column('B', dtype=float),
    column('C', dtype=str),
]))
def test_roundtrip_preserves_shape(df):
    interchange_obj = df.__dataframe__()
    result = interchange.from_dataframe(interchange_obj)
    assert result.shape == df.shape
```

**Failing input**: DataFrame with string column containing `'\ud800'` (surrogate character)

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

Output:
```
UnicodeEncodeError: 'utf-8' codec can't encode character '\ud800' in position 0: surrogates not allowed
```

## Why This Is A Bug

1. **Round-trip property violation**: Pandas allows creating DataFrames with surrogate characters in strings, but the interchange protocol cannot round-trip such data
2. **Inconsistent API**: The `__dataframe__()` method succeeds, but `from_dataframe()` fails, violating user expectations
3. **Real-world impact**: Surrogate characters can appear in real-world data (e.g., from corrupted text, certain Unicode edge cases, or data from systems that don't validate UTF-16 properly)

The bug occurs in `/pandas/core/interchange/column.py` at line 351 where the code attempts to encode strings to UTF-8 without handling surrogates:

```python
b.extend(obj.encode(encoding="utf-8"))
```

## Fix

The code should handle surrogate characters gracefully, either by:
1. Using `errors='surrogatepass'` or `errors='replace'` parameter in the encode call
2. Detecting and rejecting DataFrames with surrogates at the `__dataframe__()` stage
3. Using a different encoding that supports surrogates

Recommended fix:

```diff
--- a/pandas/core/interchange/column.py
+++ b/pandas/core/interchange/column.py
@@ -348,7 +348,7 @@
             # TODO: this for-loop is slow; can be implemented in Cython/C/C++ later
             for obj in buf:
                 if isinstance(obj, str):
-                    b.extend(obj.encode(encoding="utf-8"))
+                    b.extend(obj.encode(encoding="utf-8", errors="surrogatepass"))

                 # Convert the byte array to a NumPy array
                 data_array = np.frombuffer(b, dtype=np.uint8)
```