# Bug Report: pandas.io.json Integer Overflow with Table Orient

**Target**: `pandas.io.json.read_json` with `orient='table'`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When round-tripping a DataFrame containing uint64 values that exceed int64 maximum (2^63-1) through JSON serialization with `orient='table'`, the values silently overflow to negative int64 values, causing data corruption without warning.

## Property-Based Test

```python
import pandas as pd
import io
from hypothesis import given, strategies as st, settings
from pandas.testing import assert_frame_equal


@given(st.data())
@settings(max_examples=200)
def test_table_orient_roundtrip(data):
    num_rows = data.draw(st.integers(min_value=1, max_value=5))
    num_cols = data.draw(st.integers(min_value=1, max_value=5))

    columns = [f'col_{i}' for i in range(num_cols)]

    df_data = {}
    for col in columns:
        df_data[col] = data.draw(
            st.lists(
                st.integers(),
                min_size=num_rows,
                max_size=num_rows
            )
        )

    df = pd.DataFrame(df_data)

    json_str = df.to_json(orient='table')
    df_roundtrip = pd.read_json(io.StringIO(json_str), orient='table')

    assert_frame_equal(df, df_roundtrip, check_dtype=False)


if __name__ == "__main__":
    test_table_orient_roundtrip()
```

<details>

<summary>
**Failing input**: `[9223372036854775808]` (2^63)
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 34, in <module>
  |     test_table_orient_roundtrip()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 8, in test_table_orient_roundtrip
  |     @settings(max_examples=200)
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 28, in test_table_orient_roundtrip
    |     df_roundtrip = pd.read_json(io.StringIO(json_str), orient='table')
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/json/_json.py", line 815, in read_json
    |     return json_reader.read()
    |            ~~~~~~~~~~~~~~~~^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/json/_json.py", line 1014, in read
    |     obj = self._get_object_parser(self.data)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/json/_json.py", line 1040, in _get_object_parser
    |     obj = FrameParser(json, **kwargs).parse()
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/json/_json.py", line 1176, in parse
    |     self._parse()
    |     ~~~~~~~~~~~^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/json/_json.py", line 1416, in _parse
    |     self.obj = parse_table_schema(json, precise_float=self.precise_float)
    |                ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/json/_table_schema.py", line 360, in parse_table_schema
    |     table = ujson_loads(json, precise_float=precise_float)
    | ValueError: Value is too small
    | Falsifying example: test_table_orient_roundtrip(
    |     data=data(...),
    | )
    | Draw 1: 1
    | Draw 2: 1
    | Draw 3: [-9_223_372_036_854_775_809]
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 30, in test_table_orient_roundtrip
    |     assert_frame_equal(df, df_roundtrip, check_dtype=False)
    |     ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 1303, in assert_frame_equal
    |     assert_series_equal(
    |     ~~~~~~~~~~~~~~~~~~~^
    |         lcol,
    |         ^^^^^
    |     ...<12 lines>...
    |         check_flags=False,
    |         ^^^^^^^^^^^^^^^^^^
    |     )
    |     ^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 1021, in assert_series_equal
    |     assert_numpy_array_equal(
    |     ~~~~~~~~~~~~~~~~~~~~~~~~^
    |         lv,
    |         ^^^
    |     ...<3 lines>...
    |         index_values=left.index,
    |         ^^^^^^^^^^^^^^^^^^^^^^^^
    |     )
    |     ^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 696, in assert_numpy_array_equal
    |     _raise(left, right, err_msg)
    |     ~~~~~~^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 690, in _raise
    |     raise_assert_detail(obj, msg, left, right, index_values=index_values)
    |     ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 620, in raise_assert_detail
    |     raise AssertionError(msg)
    | AssertionError: DataFrame.iloc[:, 0] (column name="col_0") are different
    |
    | DataFrame.iloc[:, 0] (column name="col_0") values are different (100.0 %)
    | [index]: [0]
    | [left]:  [9223372036854775808]
    | [right]: [-9223372036854775808]
    | Falsifying example: test_table_orient_roundtrip(
    |     data=data(...),
    | )
    | Draw 1: 1
    | Draw 2: 1
    | Draw 3: [9_223_372_036_854_775_808]
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import io

# Create a DataFrame with a uint64 value that exceeds int64 max
# int64 max is 2^63 - 1 = 9223372036854775807
# We use 2^63 = 9223372036854775808
df = pd.DataFrame({'col': [9223372036854775808]})

print(f"Original value: {df['col'][0]}")
print(f"Original dtype: {df['col'].dtype}")

# Round-trip through JSON with table orient
json_str = df.to_json(orient='table')
print(f"\nJSON string: {json_str}")

df_roundtrip = pd.read_json(io.StringIO(json_str), orient='table')

print(f"\nRoundtrip value: {df_roundtrip['col'][0]}")
print(f"Roundtrip dtype: {df_roundtrip['col'].dtype}")
print(f"\nData corrupted: {df['col'][0] != df_roundtrip['col'][0]}")
print(f"Values differ by: {abs(df['col'][0] - df_roundtrip['col'][0])}")
```

<details>

<summary>
Output showing data corruption
</summary>
```
Original value: 9223372036854775808
Original dtype: uint64

JSON string: {"schema":{"fields":[{"name":"index","type":"integer"},{"name":"col","type":"integer"}],"primaryKey":["index"],"pandas_version":"1.4.0"},"data":[{"index":0,"col":9223372036854775808}]}

Roundtrip value: -9223372036854775808
Roundtrip dtype: int64

Data corrupted: True
Values differ by: 1.8446744073709552e+19
```
</details>

## Why This Is A Bug

This violates expected behavior in multiple critical ways:

1. **Silent data corruption**: A positive value of 9,223,372,036,854,775,808 becomes -9,223,372,036,854,775,808 with no warning or error raised. This is a catastrophic change that alters both magnitude and sign.

2. **Violates documented round-trip guarantee**: The pandas documentation states that the 'table' orient provides "the most comprehensive schema" and is specifically designed for "round-trip compatibility" between `to_json` and `read_json`. Users reasonably expect data integrity to be preserved.

3. **Schema loses type information**: The table schema uses a generic "integer" type for all integer dtypes (int8, int16, int32, int64, uint8, uint16, uint32, uint64), losing the crucial distinction between signed and unsigned types. This is visible in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/io/json/_table_schema.py:83-84` where `is_integer_dtype(x)` returns "integer" for both signed and unsigned types.

4. **Inconsistent behavior across orient modes**: Other orient modes ('split', 'records', 'index', 'columns') correctly preserve uint64 values, demonstrating that pandas is capable of handling these values properly. Only 'table' orient exhibits this bug.

5. **Actually two distinct failures**: The hypothesis test revealed two failure modes:
   - Values below int64 minimum (-9,223,372,036,854,775,809) cause a ValueError: "Value is too small"
   - Values at exactly 2^63 (9,223,372,036,854,775,808) silently overflow to negative values

## Relevant Context

The bug stems from the interaction between the Table Schema specification and pandas' implementation:

- **Table Schema Specification** (https://specs.frictionlessdata.io/table-schema/) only defines a generic "integer" type without distinguishing signed/unsigned variants
- The JSON correctly serializes the uint64 value as 9223372036854775808
- When reading back with table orient, pandas defaults all integers to int64 in `convert_json_field_to_pandas_type()` (line 200 in `_table_schema.py`)
- The ujson parser then attempts to fit the value into int64, causing overflow

Testing shows other orient modes handle the same value correctly:
- split: preserves value and uint64 dtype
- records: preserves value and uint64 dtype
- index: preserves value and uint64 dtype
- columns: preserves value and uint64 dtype
- table: corrupts to negative int64

This is particularly dangerous for applications in finance, cryptography, or scientific computing where large unsigned integers are common and sign changes are catastrophic errors.

## Proposed Fix

The table schema should distinguish between signed and unsigned integer types to preserve data integrity:

```diff
--- a/pandas/io/json/_table_schema.py
+++ b/pandas/io/json/_table_schema.py
@@ -81,8 +81,14 @@ def as_json_table_type(x: DtypeObj) -> str:
     =============== =================
     """
-    if is_integer_dtype(x):
-        return "integer"
+    if is_integer_dtype(x):
+        import numpy as np
+        if hasattr(x, 'numpy_dtype'):
+            x = x.numpy_dtype
+        if np.issubdtype(x, np.unsignedinteger):
+            return "unsignedInteger"
+        else:
+            return "integer"
     elif is_bool_dtype(x):
         return "boolean"
     elif is_numeric_dtype(x):
@@ -196,8 +202,10 @@ def convert_json_field_to_pandas_type(field) -> str | CategoricalDtype:
     typ = field["type"]
     if typ == "string":
         return field.get("extDtype", None)
     elif typ == "integer":
         return field.get("extDtype", "int64")
+    elif typ == "unsignedInteger":
+        return field.get("extDtype", "uint64")
     elif typ == "number":
         return field.get("extDtype", "float64")
     elif typ == "boolean":
```