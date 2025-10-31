# Bug Report: pandas JSON Empty DataFrame Index Dtype

**Target**: `pandas.api.typing.JsonReader` / `pandas.read_json` / `pandas.DataFrame.to_json`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When an empty DataFrame is serialized to JSON with `orient='split'` or `orient='columns'` and then deserialized, the index dtype changes from `int64` to `float64` and the index type changes from `RangeIndex` to `Index`.

## Property-Based Test

```python
import pandas as pd
from io import StringIO
from hypothesis import given, strategies as st, settings


@given(
    st.lists(st.integers(), min_size=0, max_size=20),
    st.sampled_from(['split', 'columns'])
)
@settings(max_examples=200)
def test_dataframe_json_roundtrip(values, orient):
    df = pd.DataFrame({'col': values})
    json_str = df.to_json(orient=orient)
    result = pd.read_json(StringIO(json_str), orient=orient)
    pd.testing.assert_frame_equal(result, df)
```

**Failing input**: `values=[]`, `orient='split'` (or `orient='columns'`)

## Reproducing the Bug

```python
import pandas as pd
from io import StringIO

df = pd.DataFrame({'col': []})
print(f"Original: {type(df.index).__name__}, dtype={df.index.dtype}")

json_str = df.to_json(orient='split')
result = pd.read_json(StringIO(json_str), orient='split')
print(f"Result:   {type(result.index).__name__}, dtype={result.index.dtype}")

pd.testing.assert_frame_equal(result, df)
```

Output:
```
Original: RangeIndex, dtype=int64
Result:   Index, dtype=float64
AssertionError: DataFrame.index are different
Attribute "inferred_type" are different
[left]:  floating
[right]: integer
```

## Why This Is A Bug

The round-trip property `read_json(df.to_json(orient=o), orient=o) == df` should hold for supported orient values. The JSON representation `{"columns":["col"],"index":[],"data":[]}` contains enough information to preserve the index structure, but the deserialization process incorrectly infers float64 instead of int64 for the empty index.

The root cause is in `_try_convert_data` (pandas/io/json/_json.py:1262-1287): when an empty index has object dtype, it's converted to float64 (lines 1262-1268), but the subsequent int64 coercion is skipped for empty data due to the `if len(data)` check at line 1279.

## Fix

```diff
--- a/pandas/io/json/_json.py
+++ b/pandas/io/json/_json.py
@@ -1262,10 +1262,14 @@ class Parser:
         elif is_string_dtype(data.dtype):
             # try float
             try:
                 data = data.astype("float64")
                 converted = True
             except (TypeError, ValueError):
                 pass
+        elif data.dtype == "object" and len(data) == 0:
+            # For empty object dtype, default to int64 instead of float64
+            data = data.astype("int64")
+            converted = True

         if data.dtype.kind == "f" and data.dtype != "float64":
```

Alternatively, preserve the original RangeIndex when the index is empty by detecting this case earlier in the parsing logic.