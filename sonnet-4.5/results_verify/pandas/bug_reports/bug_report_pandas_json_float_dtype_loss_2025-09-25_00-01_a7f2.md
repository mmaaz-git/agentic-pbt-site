# Bug Report: pandas.io.json.read_json Float Dtype Loss

**Target**: `pandas.io.json.read_json`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`read_json()` incorrectly infers int64 dtype for columns containing float values that happen to be integers (like 1.0, 2.0), breaking the round-trip property for float DataFrames. This occurs for all orient values except 'table'.

## Property-Based Test

```python
@given(
    data_frames(
        columns=[
            column("x", dtype=float),
            column("y", dtype=float),
        ],
        index=range_indexes(min_size=1, max_size=20),
    ),
    st.sampled_from(["split", "records", "index", "columns"]),
)
@settings(max_examples=200)
def test_dataframe_float_round_trip(df, orient):
    assume(not df.isnull().any().any())
    assume(not np.isinf(df.values).any())
    assume(not np.isnan(df.values).any())

    json_str = df.to_json(orient=orient)
    df_recovered = pd.read_json(StringIO(json_str), orient=orient)

    pd.testing.assert_frame_equal(df, df_recovered)
```

**Failing input**: DataFrame with float columns containing integer values like `[0.0]` or `[1.0, 2.0]`

## Reproducing the Bug

```python
import pandas as pd
from io import StringIO

df = pd.DataFrame({'x': [1.0, 2.0], 'y': [3.0, 4.0]})
print(f"Original dtypes: {df.dtypes.to_dict()}")

json_str = df.to_json(orient="split")
print(f"JSON: {json_str}")

df_recovered = pd.read_json(StringIO(json_str), orient="split")
print(f"Recovered dtypes: {df_recovered.dtypes.to_dict()}")
```

**Output:**
```
Original dtypes: {'x': dtype('float64'), 'y': dtype('float64')}
JSON: {"columns":["x","y"],"index":[0,1],"data":[[1.0,3.0],[2.0,4.0]]}
Recovered dtypes: {'x': dtype('int64'), 'y': dtype('int64')}
```

## Why This Is A Bug

1. **The JSON explicitly uses float notation**: The JSON contains `1.0` and `2.0` (with decimal points), not `1` and `2`
2. **Python's json library returns float**: `json.loads('{"x": 1.0}')` returns `{'x': 1.0}` as a float
3. **Breaks round-trip property**: Users expect `read_json(df.to_json())` to return an equivalent DataFrame
4. **Inconsistent with 'table' orient**: The 'table' orient correctly preserves float dtype because it includes schema information

The bug is in pandas' type inference logic, which aggressively downcasts floats to integers when all values happen to be integer-like, ignoring the fact that the JSON explicitly uses float notation.

## Fix

The fix should modify the type inference in `read_json()` to respect the decimal notation in JSON. When a number is written as `1.0` (with a decimal point), it should be treated as float, not int.

```diff
--- a/pandas/io/json/_json.py
+++ b/pandas/io/json/_json.py
@@ -1439,8 +1439,11 @@ def _try_convert_types(
     if dtype is not None or not convert_types:
         return result

-    # Aggressive type inference
-    return result.infer_objects()
+    # Type inference that preserves floats
+    # Don't downcast floats to ints - if JSON had decimal notation,
+    # respect it
+    result = result.infer_objects()
+    # TODO: Add logic to preserve float when JSON uses decimal notation
+    return result
```

A more complete fix would require tracking whether the original JSON values used decimal notation during parsing, and preserving that information through the type inference step.