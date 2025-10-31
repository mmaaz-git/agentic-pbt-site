# Bug Report: pandas.read_json Column Name Type Corruption with Numeric String Keys

**Target**: `pandas.read_json`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`pandas.read_json` incorrectly converts numeric string column names to integers when all column names are numeric strings, breaking JSON roundtrip serialization and violating the JSON specification.

## Property-Based Test

```python
from hypothesis import assume, given, settings, strategies as st
import pandas as pd
from io import StringIO


@given(st.lists(st.dictionaries(st.text(min_size=1, max_size=10), st.integers(), min_size=1, max_size=5), min_size=1, max_size=20))
@settings(max_examples=500)
def test_json_roundtrip(records):
    df = pd.DataFrame(records)
    assume(len(df.columns) > 0)

    json_str = df.to_json(orient='records')
    df_read = pd.read_json(StringIO(json_str), orient='records')

    pd.testing.assert_frame_equal(df, df_read)
```

<details>

<summary>
**Failing input**: `records=[{'0': 0}]`
</summary>
```
/home/npc/pbt/agentic-pbt/worker_/50/hypo.py:13: FutureWarning: The behavior of 'to_datetime' with 'unit' when parsing strings is deprecated. In a future version, strings will be parsed as datetime strings, matching the behavior without a 'unit'. To retain the old behavior, explicitly cast ints or floats to numeric type before calling to_datetime.
  df_read = pd.read_json(StringIO(json_str), orient='records')
/home/npc/pbt/agentic-pbt/worker_/50/hypo.py:13: FutureWarning: The behavior of 'to_datetime' with 'unit' when parsing strings is deprecated. In a future version, strings will be parsed as datetime strings, matching the behavior without a 'unit'. To retain the old behavior, explicitly cast ints or floats to numeric type before calling to_datetime.
  df_read = pd.read_json(StringIO(json_str), orient='records')
/home/npc/pbt/agentic-pbt/worker_/50/hypo.py:13: FutureWarning: The behavior of 'to_datetime' with 'unit' when parsing strings is deprecated. In a future version, strings will be parsed as datetime strings, matching the behavior without a 'unit'. To retain the old behavior, explicitly cast ints or floats to numeric type before calling to_datetime.
  df_read = pd.read_json(StringIO(json_str), orient='records')
/home/npc/pbt/agentic-pbt/worker_/50/hypo.py:13: FutureWarning: The behavior of 'to_datetime' with 'unit' when parsing strings is deprecated. In a future version, strings will be parsed as datetime strings, matching the behavior without a 'unit'. To retain the old behavior, explicitly cast ints or floats to numeric type before calling to_datetime.
  df_read = pd.read_json(StringIO(json_str), orient='records')
/home/npc/pbt/agentic-pbt/worker_/50/hypo.py:13: FutureWarning: The behavior of 'to_datetime' with 'unit' when parsing strings is deprecated. In a future version, strings will be parsed as datetime strings, matching the behavior without a 'unit'. To retain the old behavior, explicitly cast ints or floats to numeric type before calling to_datetime.
  df_read = pd.read_json(StringIO(json_str), orient='records')
/home/npc/pbt/agentic-pbt/worker_/50/hypo.py:13: FutureWarning: The behavior of 'to_datetime' with 'unit' when parsing strings is deprecated. In a future version, strings will be parsed as datetime strings, matching the behavior without a 'unit'. To retain the old behavior, explicitly cast ints or floats to numeric type before calling to_datetime.
  df_read = pd.read_json(StringIO(json_str), orient='records')
/home/npc/pbt/agentic-pbt/worker_/50/hypo.py:13: FutureWarning: The behavior of 'to_datetime' with 'unit' when parsing strings is deprecated. In a future version, strings will be parsed as datetime strings, matching the behavior without a 'unit'. To retain the old behavior, explicitly cast ints or floats to numeric type before calling to_datetime.
  df_read = pd.read_json(StringIO(json_str), orient='records')
/home/npc/pbt/agentic-pbt/worker_/50/hypo.py:13: FutureWarning: The behavior of 'to_datetime' with 'unit' when parsing strings is deprecated. In a future version, strings will be parsed as datetime strings, matching the behavior without a 'unit'. To retain the old behavior, explicitly cast ints or floats to numeric type before calling to_datetime.
  df_read = pd.read_json(StringIO(json_str), orient='records')
/home/npc/pbt/agentic-pbt/worker_/50/hypo.py:13: FutureWarning: The behavior of 'to_datetime' with 'unit' when parsing strings is deprecated. In a future version, strings will be parsed as datetime strings, matching the behavior without a 'unit'. To retain the old behavior, explicitly cast ints or floats to numeric type before calling to_datetime.
  df_read = pd.read_json(StringIO(json_str), orient='records')
/home/npc/pbt/agentic-pbt/worker_/50/hypo.py:13: FutureWarning: The behavior of 'to_datetime' with 'unit' when parsing strings is deprecated. In a future version, strings will be parsed as datetime strings, matching the behavior without a 'unit'. To retain the old behavior, explicitly cast ints or floats to numeric type before calling to_datetime.
  df_read = pd.read_json(StringIO(json_str), orient='records')
/home/npc/pbt/agentic-pbt/worker_/50/hypo.py:13: FutureWarning: The behavior of 'to_datetime' with 'unit' when parsing strings is deprecated. In a future version, strings will be parsed as datetime strings, matching the behavior without a 'unit'. To retain the old behavior, explicitly cast ints or floats to numeric type before calling to_datetime.
  df_read = pd.read_json(StringIO(json_str), orient='records')
/home/npc/pbt/agentic-pbt/worker_/50/hypo.py:13: FutureWarning: The behavior of 'to_datetime' with 'unit' when parsing strings is deprecated. In a future version, strings will be parsed as datetime strings, matching the behavior without a 'unit'. To retain the old behavior, explicitly cast ints or floats to numeric type before calling to_datetime.
  df_read = pd.read_json(StringIO(json_str), orient='records')
/home/npc/pbt/agentic-pbt/worker_/50/hypo.py:13: FutureWarning: The behavior of 'to_datetime' with 'unit' when parsing strings is deprecated. In a future version, strings will be parsed as datetime strings, matching the behavior without a 'unit'. To retain the old behavior, explicitly cast ints or floats to numeric type before calling to_datetime.
  df_read = pd.read_json(StringIO(json_str), orient='records')
/home/npc/pbt/agentic-pbt/worker_/50/hypo.py:13: FutureWarning: The behavior of 'to_datetime' with 'unit' when parsing strings is deprecated. In a future version, strings will be parsed as datetime strings, matching the behavior without a 'unit'. To retain the old behavior, explicitly cast ints or floats to numeric type before calling to_datetime.
  df_read = pd.read_json(StringIO(json_str), orient='records')
/home/npc/pbt/agentic-pbt/worker_/50/hypo.py:13: FutureWarning: The behavior of 'to_datetime' with 'unit' when parsing strings is deprecated. In a future version, strings will be parsed as datetime strings, matching the behavior without a 'unit'. To retain the old behavior, explicitly cast ints or floats to numeric type before calling to_datetime.
  df_read = pd.read_json(StringIO(json_str), orient='records')
/home/npc/pbt/agentic-pbt/worker_/50/hypo.py:13: FutureWarning: The behavior of 'to_datetime' with 'unit' when parsing strings is deprecated. In a future version, strings will be parsed as datetime strings, matching the behavior without a 'unit'. To retain the old behavior, explicitly cast ints or floats to numeric type before calling to_datetime.
  df_read = pd.read_json(StringIO(json_str), orient='records')
/home/npc/pbt/agentic-pbt/worker_/50/hypo.py:13: FutureWarning: The behavior of 'to_datetime' with 'unit' when parsing strings is deprecated. In a future version, strings will be parsed as datetime strings, matching the behavior without a 'unit'. To retain the old behavior, explicitly cast ints or floats to numeric type before calling to_datetime.
  df_read = pd.read_json(StringIO(json_str), orient='records')
/home/npc/pbt/agentic-pbt/worker_/50/hypo.py:13: FutureWarning: The behavior of 'to_datetime' with 'unit' when parsing strings is deprecated. In a future version, strings will be parsed as datetime strings, matching the behavior without a 'unit'. To retain the old behavior, explicitly cast ints or floats to numeric type before calling to_datetime.
  df_read = pd.read_json(StringIO(json_str), orient='records')
/home/npc/pbt/agentic-pbt/worker_/50/hypo.py:13: FutureWarning: The behavior of 'to_datetime' with 'unit' when parsing strings is deprecated. In a future version, strings will be parsed as datetime strings, matching the behavior without a 'unit'. To retain the old behavior, explicitly cast ints or floats to numeric type before calling to_datetime.
  df_read = pd.read_json(StringIO(json_str), orient='records')
/home/npc/pbt/agentic-pbt/worker_/50/hypo.py:13: FutureWarning: The behavior of 'to_datetime' with 'unit' when parsing strings is deprecated. In a future version, strings will be parsed as datetime strings, matching the behavior without a 'unit'. To retain the old behavior, explicitly cast ints or floats to numeric type before calling to_datetime.
  df_read = pd.read_json(StringIO(json_str), orient='records')
/home/npc/pbt/agentic-pbt/worker_/50/hypo.py:13: FutureWarning: The behavior of 'to_datetime' with 'unit' when parsing strings is deprecated. In a future version, strings will be parsed as datetime strings, matching the behavior without a 'unit'. To retain the old behavior, explicitly cast ints or floats to numeric type before calling to_datetime.
  df_read = pd.read_json(StringIO(json_str), orient='records')
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 19, in <module>
  |     test_json_roundtrip()
  |     ~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 7, in test_json_roundtrip
  |     @settings(max_examples=500)
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 13, in test_json_roundtrip
    |     df_read = pd.read_json(StringIO(json_str), orient='records')
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
    | Falsifying example: test_json_roundtrip(
    |     records=[{'0': -9_223_372_036_854_775_809}],
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/construction.py:604
    |         /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/construction.py:605
    |         /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/internals/construction.py:1045
    |         /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/internals/construction.py:1047
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 15, in test_json_roundtrip
    |     pd.testing.assert_frame_equal(df, df_read)
    |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 1264, in assert_frame_equal
    |     assert_index_equal(
    |     ~~~~~~~~~~~~~~~~~~^
    |         left.columns,
    |         ^^^^^^^^^^^^^
    |     ...<8 lines>...
    |         obj=f"{obj}.columns",
    |         ^^^^^^^^^^^^^^^^^^^^^
    |     )
    |     ^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 253, in assert_index_equal
    |     _check_types(left, right, obj=obj)
    |     ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 236, in _check_types
    |     assert_attr_equal("inferred_type", left, right, obj=obj)
    |     ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 421, in assert_attr_equal
    |     raise_assert_detail(obj, msg, left_attr, right_attr)
    |     ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 620, in raise_assert_detail
    |     raise AssertionError(msg)
    | AssertionError: DataFrame.columns are different
    |
    | Attribute "inferred_type" are different
    | [left]:  string
    | [right]: integer
    | Falsifying example: test_json_roundtrip(
    |     records=[{'0': 0}],
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py:420
    |         /home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py:610
    |         /home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py:612
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
import pandas as pd
from io import StringIO

# Create a DataFrame with numeric string column names
df = pd.DataFrame({'0': [1, 2], '1': [3, 4]})
print('Original DataFrame:')
print(df)
print('Original columns:', df.columns.tolist())
print('Original column types:', [type(c).__name__ for c in df.columns])
print()

# Convert to JSON
json_str = df.to_json(orient='records')
print('JSON representation:')
print(json_str)
print()

# Read back from JSON
df_back = pd.read_json(StringIO(json_str), orient='records')
print('DataFrame after roundtrip:')
print(df_back)
print('After roundtrip columns:', df_back.columns.tolist())
print('After roundtrip column types:', [type(c).__name__ for c in df_back.columns])
print()

# Verify the issue
print('Column names match?', df.columns.tolist() == df_back.columns.tolist())
print('Expected columns: ["0", "1"]')
print('Actual columns after roundtrip:', df_back.columns.tolist())

# Show the inconsistency with mixed column types
print('\n--- Demonstrating inconsistent behavior ---')
df_mixed = pd.DataFrame({'0': [1], '1': [2], 'name': ['test']})
json_mixed = df_mixed.to_json(orient='records')
df_mixed_back = pd.read_json(StringIO(json_mixed), orient='records')
print('Mixed columns original:', df_mixed.columns.tolist())
print('Mixed columns after roundtrip:', df_mixed_back.columns.tolist())
print('Types after roundtrip:', [type(c).__name__ for c in df_mixed_back.columns])
```

<details>

<summary>
Column type conversion breaks JSON roundtrip
</summary>
```
/home/npc/pbt/agentic-pbt/worker_/50/repo.py:35: FutureWarning: The behavior of 'to_datetime' with 'unit' when parsing strings is deprecated. In a future version, strings will be parsed as datetime strings, matching the behavior without a 'unit'. To retain the old behavior, explicitly cast ints or floats to numeric type before calling to_datetime.
  df_mixed_back = pd.read_json(StringIO(json_mixed), orient='records')
/home/npc/pbt/agentic-pbt/worker_/50/repo.py:35: FutureWarning: The behavior of 'to_datetime' with 'unit' when parsing strings is deprecated. In a future version, strings will be parsed as datetime strings, matching the behavior without a 'unit'. To retain the old behavior, explicitly cast ints or floats to numeric type before calling to_datetime.
  df_mixed_back = pd.read_json(StringIO(json_mixed), orient='records')
/home/npc/pbt/agentic-pbt/worker_/50/repo.py:35: FutureWarning: The behavior of 'to_datetime' with 'unit' when parsing strings is deprecated. In a future version, strings will be parsed as datetime strings, matching the behavior without a 'unit'. To retain the old behavior, explicitly cast ints or floats to numeric type before calling to_datetime.
  df_mixed_back = pd.read_json(StringIO(json_mixed), orient='records')
Original DataFrame:
   0  1
0  1  3
1  2  4
Original columns: ['0', '1']
Original column types: ['str', 'str']

JSON representation:
[{"0":1,"1":3},{"0":2,"1":4}]

DataFrame after roundtrip:
   0  1
0  1  3
1  2  4
After roundtrip columns: [0, 1]
After roundtrip column types: ['int', 'int']

Column names match? False
Expected columns: ["0", "1"]
Actual columns after roundtrip: [0, 1]

--- Demonstrating inconsistent behavior ---
Mixed columns original: ['0', '1', 'name']
Mixed columns after roundtrip: ['0', '1', 'name']
Types after roundtrip: ['str', 'str', 'str']
```
</details>

## Why This Is A Bug

This behavior violates several fundamental principles and expectations:

1. **JSON Specification Violation**: According to RFC 7159, JSON object keys are always strings. When `read_json` converts string keys like `"0"` to integer `0`, it's changing the fundamental data type in violation of the specification. The JSON contains `{"0":1,"1":3}` with string keys, not integer keys.

2. **Broken Roundtrip Contract**: A fundamental expectation of serialization is that it should be reversible - `df.to_json()` followed by `read_json()` should return an equivalent DataFrame. This bug breaks that contract, as the column types change from strings to integers.

3. **Inconsistent Behavior**: The conversion only happens when ALL column names are numeric strings. If there's even one non-numeric column name, all columns remain as strings. This inconsistency makes the behavior unpredictable and data-dependent:
   - `['0', '1']` → `[0, 1]` (converted)
   - `['0', '1', 'name']` → `['0', '1', 'name']` (preserved)

4. **Silent Data Corruption**: The conversion happens without any warning or error, leading to subtle bugs in downstream code that expects string column names. This is particularly problematic for applications that use numeric-looking string identifiers.

5. **Real-World Impact**: This affects common use cases such as:
   - DataFrames with year-based columns ('2020', '2021', '2022')
   - ID-based columns ('0', '1', '2', '3')
   - Any system using numeric string identifiers

## Relevant Context

The issue occurs in pandas version 2.3.2 and is triggered when using `orient='records'` with `read_json()`. The bug appears to be in the column name inference logic that inappropriately applies type conversion to column names when they all appear numeric.

Documentation references:
- pandas.read_json: https://pandas.pydata.org/docs/reference/api/pandas.read_json.html
- JSON RFC 7159: https://www.rfc-editor.org/rfc/rfc7159.html#section-4

The behavior is particularly problematic because:
- There's no parameter to control column name type inference (unlike value type inference with `dtype` parameter)
- The inconsistency based on data content makes it hard to predict
- It silently breaks compatibility with systems expecting string column names

## Proposed Fix

The bug appears to be in the DataFrame construction after JSON parsing. When `orient='records'`, pandas should preserve string column names from the JSON keys. A potential fix would be to ensure column names remain as strings when constructing the DataFrame from the parsed JSON dictionary:

```diff
--- a/pandas/io/json/_json.py
+++ b/pandas/io/json/_json.py
@@ -1417,7 +1417,9 @@ class FrameParser(Parser):
         else:
             self.obj = DataFrame(
-                ujson_loads(json, precise_float=self.precise_float), dtype=None
+                ujson_loads(json, precise_float=self.precise_float),
+                dtype=None,
+                columns=pd.Index(data[0].keys(), dtype=object) if data else None
             )
```

Alternative approaches:
1. Add a parameter like `preserve_column_types=True` to control this behavior
2. Document the current behavior if it's intentional (though this seems unlikely given the inconsistency)
3. Apply consistent behavior regardless of whether all columns are numeric or mixed