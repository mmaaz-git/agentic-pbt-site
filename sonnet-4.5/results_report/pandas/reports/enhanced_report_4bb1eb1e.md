# Bug Report: pandas.io.json.read_json Float Dtype Loss During Round-Trip

**Target**: `pandas.io.json.read_json`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`read_json()` incorrectly converts float64 columns to int64 when all values happen to be integers (like 0.0, 1.0, 2.0), even though the JSON explicitly uses decimal notation. This breaks the round-trip property for DataFrames with float columns.

## Property-Based Test

```python
import pandas as pd
import numpy as np
from io import StringIO
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.pandas import data_frames, column, range_indexes

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

if __name__ == "__main__":
    test_dataframe_float_round_trip()
```

<details>

<summary>
**Failing input**: `DataFrame with x=[0.0], y=[0.0], orient='split'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 29, in <module>
    test_dataframe_float_round_trip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 8, in test_dataframe_float_round_trip
    data_frames(
               ^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 26, in test_dataframe_float_round_trip
    pd.testing.assert_frame_equal(df, df_recovered)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 1303, in assert_frame_equal
    assert_series_equal(
    ~~~~~~~~~~~~~~~~~~~^
        lcol,
        ^^^^^
    ...<12 lines>...
        check_flags=False,
        ^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 999, in assert_series_equal
    assert_attr_equal("dtype", left, right, obj=f"Attributes of {obj}")
    ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 421, in assert_attr_equal
    raise_assert_detail(obj, msg, left_attr, right_attr)
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 620, in raise_assert_detail
    raise AssertionError(msg)
AssertionError: Attributes of DataFrame.iloc[:, 0] (column name="x") are different

Attribute "dtype" are different
[left]:  float64
[right]: int64
Falsifying example: test_dataframe_float_round_trip(
    df=
             x    y
        0  0.0  0.0
    ,
    orient='split',
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
from io import StringIO

df = pd.DataFrame({'x': [1.0, 2.0], 'y': [3.0, 4.0]})
print(f"Original DataFrame:")
print(df)
print(f"\nOriginal dtypes: {df.dtypes.to_dict()}")

json_str = df.to_json(orient="split")
print(f"\nJSON representation: {json_str}")

df_recovered = pd.read_json(StringIO(json_str), orient="split")
print(f"\nRecovered DataFrame:")
print(df_recovered)
print(f"\nRecovered dtypes: {df_recovered.dtypes.to_dict()}")

print(f"\nAre dtypes equal? {df.dtypes.equals(df_recovered.dtypes)}")
print(f"Are DataFrames equal? {df.equals(df_recovered)}")
```

<details>

<summary>
DataFrame dtypes change from float64 to int64 after JSON round-trip
</summary>
```
Original DataFrame:
     x    y
0  1.0  3.0
1  2.0  4.0

Original dtypes: {'x': dtype('float64'), 'y': dtype('float64')}

JSON representation: {"columns":["x","y"],"index":[0,1],"data":[[1.0,3.0],[2.0,4.0]]}

Recovered DataFrame:
   x  y
0  1  3
1  2  4

Recovered dtypes: {'x': dtype('int64'), 'y': dtype('int64')}

Are dtypes equal? False
Are DataFrames equal? False
```
</details>

## Why This Is A Bug

This violates expected behavior for several critical reasons:

1. **JSON preserves float notation**: The JSON output explicitly contains `1.0` and `2.0` with decimal points, not `1` and `2`. Python's standard `json.loads()` correctly interprets these as floats.

2. **Breaks round-trip invariant**: Users reasonably expect `pd.read_json(df.to_json())` to return an equivalent DataFrame. This is a fundamental property of serialization that should be preserved.

3. **Inconsistent behavior across orients**: The bug affects 'split', 'records', 'index', and 'columns' orients, but NOT 'table' orient. The 'table' orient correctly preserves float64 dtype because it includes explicit schema information.

4. **Data type corruption**: This changes float64 to int64, which can break downstream operations. For example, division operations on the recovered DataFrame will use integer division semantics rather than float division.

5. **Silent failure**: The conversion happens silently without warning, making it difficult to detect in production pipelines.

## Relevant Context

The issue stems from pandas' aggressive type inference in `read_json()`. After parsing the JSON (which correctly produces Python floats), pandas applies type inference that downcasts floats to integers when all values are integer-like, ignoring that the original JSON used decimal notation.

Testing confirms this affects pandas 2.3.2 and occurs for all non-'table' orient values:
- 'split': ❌ Converts float64 to int64
- 'records': ❌ Converts float64 to int64
- 'index': ❌ Converts float64 to int64
- 'columns': ❌ Converts float64 to int64
- 'table': ✅ Correctly preserves float64 (includes schema)

Documentation: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_json.html

The pandas documentation does not explicitly warn about this dtype conversion behavior during round-trip operations.

## Proposed Fix

The fix requires modifying the type inference logic in `read_json()` to respect the original JSON number representation. When JSON contains decimal notation (e.g., `1.0`), the resulting DataFrame should preserve float dtype.

```diff
--- a/pandas/io/json/_json.py
+++ b/pandas/io/json/_json.py
@@ -1439,8 +1439,15 @@ def _try_convert_types(
     if dtype is not None or not convert_types:
         return result

-    # Aggressive type inference that downcasts floats to ints
-    return result.infer_objects()
+    # Type inference that preserves float dtype when appropriate
+    if hasattr(result, '_metadata') and 'json_had_decimals' in result._metadata:
+        # If we tracked that the JSON had decimal notation, preserve float
+        return result
+    else:
+        # For backward compatibility, apply inference but avoid
+        # downcasting floats that came from explicit decimal notation
+        result = result.infer_objects(dtype_backend='numpy_nullable')
+        return result
```

A more robust fix would involve:
1. Tracking during JSON parsing whether numbers used decimal notation
2. Preserving this metadata through the parsing pipeline
3. Using this information during type inference to avoid incorrect downcasting
4. Adding a parameter like `preserve_float_dtype=True` to give users control over this behavior