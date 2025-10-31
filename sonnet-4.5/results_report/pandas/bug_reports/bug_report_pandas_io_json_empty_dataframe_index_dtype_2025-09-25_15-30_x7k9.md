# Bug Report: pandas.io.json Empty DataFrame Index Dtype Loss

**Target**: `pandas.io.json.read_json` and `pandas.io.json.to_json`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When round-tripping an empty DataFrame through JSON using `orient='split'` or `orient='columns'`, the index dtype is incorrectly changed from `int64` to `float64`, violating the expected round-trip property.

## Property-Based Test

```python
import io
import pandas as pd
from hypothesis import given, settings
from hypothesis.extra.pandas import column, data_frames, range_indexes


@settings(max_examples=500)
@given(
    df=data_frames(
        columns=[
            column("a", dtype=int),
            column("b", dtype=float),
        ],
        index=range_indexes(min_size=0, max_size=20),
    )
)
def test_json_roundtrip_split_preserves_index_dtype(df):
    json_str = df.to_json(orient="split")
    result = pd.read_json(io.StringIO(json_str), orient="split")
    assert df.index.dtype == result.index.dtype
```

**Failing input**: An empty DataFrame with int64 index

## Reproducing the Bug

```python
import io
import pandas as pd

df = pd.DataFrame({"a": [], "b": []})
df["a"] = df["a"].astype(int)
df["b"] = df["b"].astype(float)

print(f"Original index dtype: {df.index.dtype}")

json_str = df.to_json(orient="split")
result = pd.read_json(io.StringIO(json_str), orient="split")

print(f"Result index dtype: {result.index.dtype}")
print(f"JSON: {json_str}")
```

Output:
```
Original index dtype: int64
Result index dtype: float64
JSON: {"columns":["a","b"],"index":[],"data":[]}
```

## Why This Is A Bug

The pandas documentation and examples demonstrate that `read_json` should be able to reconstruct DataFrames from their JSON representation. This round-trip property is broken for empty DataFrames with `orient='split'` and `orient='columns'`.

The bug occurs because the code at `pandas/io/json/_json.py:1298-1300` only preserves index dtype when the index is non-empty:

```python
# if we have an index, we want to preserve dtypes
if name == "index" and len(data):
    if self.orient == "split":
        return data, False
```

For empty indexes, the dtype is not preserved, and pandas defaults to `float64`.

**Affected orients**: `split`, `columns`
**Unaffected orients**: `index`, `records`, `values`

## Fix

```diff
--- a/pandas/io/json/_json.py
+++ b/pandas/io/json/_json.py
@@ -1295,7 +1295,7 @@ class Parser:
                 pass

         # if we have an index, we want to preserve dtypes
-        if name == "index" and len(data):
+        if name == "index":
             if self.orient == "split":
                 return data, False
```

This fix ensures that the index dtype is preserved even for empty DataFrames when using `orient='split'`. However, this may not fully address the issue for `orient='columns'`, which would need additional investigation.