# Bug Report: pandas.io.json Integer Overflow on Round-Trip

**Target**: `pandas.io.json.read_json` and `pandas.io.json.to_json`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When a DataFrame contains integer values outside the int64/uint64 range, `to_json()` successfully serializes them to JSON, but `read_json()` crashes with `ValueError: Value is too small` or `ValueError: Value is too big!` when attempting to deserialize, violating the documented round-trip property.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd
from io import StringIO

@given(
    data=st.lists(
        st.dictionaries(
            st.text(min_size=1, max_size=10),
            st.integers(),
            min_size=1,
            max_size=5,
        ),
        min_size=1,
        max_size=20,
    ),
    orient=st.sampled_from(['records', 'index', 'columns', 'values', 'split']),
)
@settings(max_examples=100)
def test_read_json_to_json_roundtrip_dataframe(data, orient):
    df = pd.DataFrame(data)
    json_str = df.to_json(orient=orient)
    df_back = pd.read_json(StringIO(json_str), orient=orient)

if __name__ == "__main__":
    test_read_json_to_json_roundtrip_dataframe()
```

<details>

<summary>
**Failing input**: `data=[{'0': -9223372036854775809}], orient='split'`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 25, in <module>
  |     test_read_json_to_json_roundtrip_dataframe()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 6, in test_read_json_to_json_roundtrip_dataframe
  |     data=st.lists(
  |                ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 4 distinct failures. (4 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 22, in test_read_json_to_json_roundtrip_dataframe
    |     df_back = pd.read_json(StringIO(json_str), orient=orient)
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
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/json/_json.py", line 1397, in _parse
    |     for k, v in ujson_loads(json, precise_float=self.precise_float).items()
    |                 ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    | ValueError: Value is too small
    | Falsifying example: test_read_json_to_json_roundtrip_dataframe(
    |     data=[{'0': -9_223_372_036_854_775_809}],
    |     orient='split',
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 22, in test_read_json_to_json_roundtrip_dataframe
    |     df_back = pd.read_json(StringIO(json_str), orient=orient)
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
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/json/_json.py", line 1392, in _parse
    |     ujson_loads(json, precise_float=self.precise_float), dtype=None
    |     ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    | ValueError: Value is too small
    | Falsifying example: test_read_json_to_json_roundtrip_dataframe(
    |     data=[{'0': -9_223_372_036_854_775_809}],
    |     orient='columns',
    | )
    +---------------- 3 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 22, in test_read_json_to_json_roundtrip_dataframe
    |     df_back = pd.read_json(StringIO(json_str), orient=orient)
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
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/json/_json.py", line 1411, in _parse
    |     ujson_loads(json, precise_float=self.precise_float),
    |     ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    | ValueError: Value is too small
    | Falsifying example: test_read_json_to_json_roundtrip_dataframe(
    |     data=[{'0': -9_223_372_036_854_775_809}],
    |     orient='index',
    | )
    +---------------- 4 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 22, in test_read_json_to_json_roundtrip_dataframe
    |     df_back = pd.read_json(StringIO(json_str), orient=orient)
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
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/json/_json.py", line 1419, in _parse
    |     ujson_loads(json, precise_float=self.precise_float), dtype=None
    |     ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    | ValueError: Value is too small
    | Falsifying example: test_read_json_to_json_roundtrip_dataframe(
    |     data=[{'0': -9_223_372_036_854_775_809}],
    |     orient='records',
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
import pandas as pd
from io import StringIO

# Test with integer value outside int64 range (one less than int64 minimum)
df = pd.DataFrame([{'col': -9223372036854775809}])
print("Original DataFrame:")
print(df)
print(f"DataFrame dtype: {df['col'].dtype}")
print(f"Value type: {type(df['col'].iloc[0])}")

# Serialize to JSON (this works)
json_str = df.to_json(orient='split')
print(f"\nSerialized JSON string:")
print(json_str)

# Try to deserialize (this crashes)
print("\nAttempting to read JSON back...")
try:
    df_back = pd.read_json(StringIO(json_str), orient='split')
    print("Successfully read back:")
    print(df_back)
except ValueError as e:
    print(f"ERROR: {e}")
```

<details>

<summary>
ValueError: Value is too small
</summary>
```
Original DataFrame:
                   col
0  -9223372036854775809
DataFrame dtype: object
Value type: <class 'int'>

Serialized JSON string:
{"columns":["col"],"index":[0],"data":[[-9223372036854775809]]}

Attempting to read JSON back...
ERROR: Value is too small
```
</details>

## Why This Is A Bug

This violates the expected round-trip behavior documented in pandas. The `read_json` documentation explicitly states: "Compatible JSON strings can be produced by `to_json()` with a corresponding orient value." This strongly implies that data written with `to_json()` should be readable with `read_json()`.

The asymmetric behavior creates a data integrity issue where:
1. Users can create valid DataFrames with Python's arbitrary-precision integers
2. These DataFrames serialize successfully to JSON with `to_json()`
3. The same JSON fails to deserialize with `read_json()`, causing data loss

The bug occurs because pandas uses the `ujson` library for JSON operations. While `ujson_dumps` can serialize Python integers of any size to JSON, `ujson_loads` has hard limits on integer size during deserialization:
- Minimum value: -9223372036854775808 (int64 min)
- Maximum value: 18446744073709551615 (uint64 max)

Values outside this range trigger `ValueError: Value is too small` or `ValueError: Value is too big!` errors.

## Relevant Context

The issue lies in the asymmetric capabilities of the underlying ujson library used by pandas:
- Location: `/pandas/io/json/_json.py`
- `ujson_dumps` (line 263): Successfully serializes any Python integer
- `ujson_loads` (lines 1362, 1392, 1397, 1411, 1419): Fails on integers outside int64/uint64 range

Testing shows the exact boundaries:
- Values < -9223372036854775808: "Value is too small"
- Values in [-9223372036854775808, 18446744073709551615]: Success
- Values > 18446744073709551615: "Value is too big!"

The DataFrame stores these large integers with dtype `object` (Python integers), not as numpy int64, which is why serialization succeeds. However, ujson's C implementation cannot handle these during deserialization.

Documentation links:
- [pandas.DataFrame.to_json](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_json.html)
- [pandas.read_json](https://pandas.pydata.org/docs/reference/api/pandas.read_json.html)

## Proposed Fix

The cleanest fix is to validate integer ranges during serialization, failing fast with a clear error message rather than allowing silent data corruption:

```diff
--- a/pandas/io/json/_json.py
+++ b/pandas/io/json/_json.py
@@ -259,6 +259,25 @@ class Writer(ABC):
         self.indent = indent

     def write(self) -> str:
+        # Validate integer values are within ujson's supported range
+        import numpy as np
+        if self.obj_to_write is not None:
+            def check_int_range(val):
+                if isinstance(val, (int, np.integer)) and not isinstance(val, (bool, np.bool_)):
+                    if not (-9223372036854775808 <= val <= 18446744073709551615):
+                        raise ValueError(
+                            f"Integer value {val} is outside the range that can be "
+                            f"reliably round-tripped through JSON (int64 min to uint64 max). "
+                            f"Consider converting to float or string before serialization."
+                        )
+
+            # Check all values in the object
+            if hasattr(self.obj_to_write, 'values'):
+                # DataFrame or Series
+                np.vectorize(check_int_range)(self.obj_to_write.values)
+            elif isinstance(self.obj_to_write, dict):
+                # Dictionary representation
+                for v in self.obj_to_write.values():
+                    np.vectorize(check_int_range)(v)
         iso_dates = self.date_format == "iso"
         return ujson_dumps(
             self.obj_to_write,
```