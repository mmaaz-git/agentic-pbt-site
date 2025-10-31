# Bug Report: pandas.io.json JSON Parse Error and Data Corruption for Large Integers

**Target**: `pandas.io.json.read_json`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

pandas JSON round-trip fails for integers outside int64 range, causing either parse errors for values below int64 minimum or silent data corruption for values above int64 maximum, breaking the fundamental expectation that `to_json()` and `read_json()` should preserve data integrity.

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
**Failing input**: `-9223372036854775809` (int64 min - 1) and `9223372036854775808` (int64 max + 1)
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 34, in <module>
  |     test_table_orient_roundtrip()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 8, in test_table_orient_roundtrip
  |     @settings(max_examples=200)
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 28, in test_table_orient_roundtrip
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
    |   File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 30, in test_table_orient_roundtrip
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

# Test with value that's below int64 minimum (-2^63)
# int64 min is -9223372036854775808
# Testing with -9223372036854775809 (int64 min - 1)
df = pd.DataFrame({'col': [-9223372036854775809]})

print("Testing pandas JSON round-trip for large negative integer (-9223372036854775809)")
print("="*70)

for orient in ['split', 'records', 'index', 'columns', 'values', 'table']:
    print(f"\nTesting orient='{orient}':")
    try:
        # Attempt to convert to JSON
        json_str = df.to_json(orient=orient)
        print(f"  to_json: Success")
        print(f"  JSON output: {json_str[:100]}{'...' if len(json_str) > 100 else ''}")

        # Attempt to read back the JSON
        df_roundtrip = pd.read_json(io.StringIO(json_str), orient=orient)
        print(f"  read_json: Success")
        print(f"  Roundtrip value: {df_roundtrip['col'][0]}")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")
```

<details>

<summary>
All orients fail with "ValueError: Value is too small"
</summary>
```
Testing pandas JSON round-trip for large negative integer (-9223372036854775809)
======================================================================

Testing orient='split':
  to_json: Success
  JSON output: {"columns":["col"],"index":[0],"data":[[-9223372036854775809]]}
  Error: ValueError: Value is too small

Testing orient='records':
  to_json: Success
  JSON output: [{"col":-9223372036854775809}]
  Error: ValueError: Value is too small

Testing orient='index':
  to_json: Success
  JSON output: {"0":{"col":-9223372036854775809}}
  Error: ValueError: Value is too small

Testing orient='columns':
  to_json: Success
  JSON output: {"col":{"0":-9223372036854775809}}
  Error: ValueError: Value is too small

Testing orient='values':
  to_json: Success
  JSON output: [[-9223372036854775809]]
  Error: ValueError: Value is too small

Testing orient='table':
  to_json: Success
  JSON output: {"schema":{"fields":[{"name":"index","type":"integer"},{"name":"col","type":"string"}],"primaryKey":...
  Error: ValueError: Value is too small
```
</details>

## Why This Is A Bug

This violates fundamental expectations of JSON round-trip operations in pandas:

1. **Silent data corruption**: Values above int64 maximum (2^63-1) are silently converted to negative values. For example, 9223372036854775808 becomes -9223372036854775808, causing data loss without warning.

2. **Asymmetric behavior**: `to_json()` successfully serializes integers outside int64 range for all orient options, producing valid JSON. However, `read_json()` fails to parse this same JSON, breaking the round-trip contract that data should be preserved.

3. **Misleading error messages**: The error "Value is too small" from ujson doesn't indicate the actual problem (integer overflow) or provide guidance on resolution.

4. **Inconsistent with Python's integer support**: Python natively supports arbitrary precision integers, and standard `json.loads()` can handle these values correctly. The limitation comes from pandas' use of ujson for performance.

5. **No documentation**: The pandas documentation doesn't warn users about this int64 limitation in JSON operations, leading to unexpected failures in production.

## Relevant Context

The issue stems from pandas' use of the ujson library for performance optimization. ujson is a fast JSON parser written in C, but it's limited to int64 range (-2^63 to 2^63-1). When values exceed this range:

- Values below int64 min raise `ValueError: Value is too small`
- Values above int64 max undergo integer overflow and wrap around to negative values

The problem occurs in multiple locations:
- `pandas/io/json/_json.py` lines 1362, 1392, 1397, 1411, 1419 (various ujson_loads calls)
- `pandas/io/json/_table_schema.py` line 360 (parse_table_schema)

Python's standard `json` module handles these values correctly but is slower. A robust solution would detect ujson failures and fall back to standard json parsing.

## Proposed Fix

```diff
--- a/pandas/io/json/_json.py
+++ b/pandas/io/json/_json.py
@@ -1359,7 +1359,14 @@ class Parser:
     def _parse(self):
         if self.orient == "split":
-            data = ujson_loads(self.json, precise_float=self.precise_float)
+            try:
+                data = ujson_loads(self.json, precise_float=self.precise_float)
+            except ValueError as e:
+                if "too small" in str(e).lower() or "too big" in str(e).lower():
+                    # Fall back to standard json for large integers
+                    import json
+                    data = json.loads(self.json)
+                else:
+                    raise
             decoded = {str(k): v for k, v in data.items()}
             self.obj = Series(**decoded)
         elif self.orient == "records" or self.orient == "values":

--- a/pandas/io/json/_table_schema.py
+++ b/pandas/io/json/_table_schema.py
@@ -357,7 +357,14 @@ def parse_table_schema(json: str, precise_float: bool) -> DataFrame:
         If the JSON table schema is not in a supported format.
     """
-    table = ujson_loads(json, precise_float=precise_float)
+    try:
+        table = ujson_loads(json, precise_float=precise_float)
+    except ValueError as e:
+        if "too small" in str(e).lower() or "too big" in str(e).lower():
+            # Fall back to standard json for large integers
+            import json as std_json
+            table = std_json.loads(json)
+        else:
+            raise
     schema = table["schema"]
     df = DataFrame(table["data"])[
         [col["name"] for col in schema["fields"]]
```