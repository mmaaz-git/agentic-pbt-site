# Bug Report: pandas.read_json Integer Underflow Asymmetric Handling

**Target**: `pandas.io.json._json.read_json`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`pandas.read_json()` exhibits asymmetric handling of integers outside the int64 range: it silently converts values above int64_max to uint64, but crashes with a ValueError for values below int64_min, creating an inconsistent and undocumented API behavior.

## Property-Based Test

```python
import pandas as pd
import json
import io
from hypothesis import given, strategies as st, settings, example


@given(st.lists(st.dictionaries(
    keys=st.text(min_size=1, max_size=20),
    values=st.one_of(st.integers(), st.floats(allow_nan=False, allow_infinity=False), st.text(max_size=50), st.none())
), min_size=1, max_size=50))
@example([{'0': -9223372036854775809}])  # The failing case from the bug report
@settings(max_examples=10)  # Run a reasonable number of tests
def test_jsonreader_basic_parsing(data):
    json_str = json.dumps(data)
    json_bytes = json_str.encode('utf-8')
    json_io = io.BytesIO(json_bytes)

    reader = pd.read_json(json_io, lines=False)

    assert len(reader) == len(data)


if __name__ == "__main__":
    # Run the test
    test_jsonreader_basic_parsing()
```

<details>

<summary>
**Failing input**: `data=[{'0': -9223372036854775809}]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 25, in <module>
    test_jsonreader_basic_parsing()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 8, in test_jsonreader_basic_parsing
    keys=st.text(min_size=1, max_size=20),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 18, in test_jsonreader_basic_parsing
    reader = pd.read_json(json_io, lines=False)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/json/_json.py", line 815, in read_json
    return json_reader.read()
           ~~~~~~~~~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/json/_json.py", line 1014, in read
    obj = self._get_object_parser(self.data)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/json/_json.py", line 1040, in _get_object_parser
    obj = FrameParser(json, **kwargs).parse()
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/json/_json.py", line 1176, in parse
    self._parse()
    ~~~~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/json/_json.py", line 1392, in _parse
    ujson_loads(json, precise_float=self.precise_float), dtype=None
    ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: Value is too small
Falsifying explicit example: test_jsonreader_basic_parsing(
    data=[{'0': -9_223_372_036_854_775_809}],
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import json
import io

# Test case for value below int64_min
int64_min = -2**63
value_below_min = int64_min - 1  # -9223372036854775809
data = [{'key': value_below_min}]
json_str = json.dumps(data)
json_io = io.BytesIO(json_str.encode('utf-8'))

print(f"Testing with value below int64_min: {value_below_min}")
print(f"JSON string: {json_str}")

try:
    result = pd.read_json(json_io, lines=False)
    print(f"Success! Result:\n{result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\n" + "="*50 + "\n")

# Test case for value above int64_max
int64_max = 2**63 - 1
value_above_max = int64_max + 1  # 9223372036854775808
data = [{'key': value_above_max}]
json_str = json.dumps(data)
json_io = io.BytesIO(json_str.encode('utf-8'))

print(f"Testing with value above int64_max: {value_above_max}")
print(f"JSON string: {json_str}")

try:
    result = pd.read_json(json_io, lines=False)
    print(f"Success! Result:\n{result}")
    print(f"Data type of 'key' column: {result['key'].dtype}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
```

<details>

<summary>
ValueError for underflow but silent conversion for overflow
</summary>
```
Testing with value below int64_min: -9223372036854775809
JSON string: [{"key": -9223372036854775809}]
Error: ValueError: Value is too small

==================================================

Testing with value above int64_max: 9223372036854775808
JSON string: [{"key": 9223372036854775808}]
Success! Result:
                  key
0  9223372036854775808
Data type of 'key' column: uint64
```
</details>

## Why This Is A Bug

This violates expected behavior in several critical ways:

1. **Asymmetric API behavior**: The function silently converts positive overflow (values > 2^63-1) to uint64, but crashes with ValueError for negative underflow (values < -2^63). This inconsistency violates the principle of least surprise and makes the API unpredictable.

2. **Both values are valid JSON**: The JSON specification (RFC 8259) allows arbitrary precision integers. Python's standard `json.dumps()` successfully serializes both -9223372036854775809 and 9223372036854775808, confirming they are valid JSON values.

3. **Undocumented behavior**: The pandas documentation for `read_json()` does not mention:
   - Any limitations on integer ranges
   - That values below int64_min will raise ValueError
   - That values above int64_max will be silently converted to uint64
   - The asymmetric nature of overflow/underflow handling

4. **Silent data type change vs crash**: For positive overflow, the function silently changes the data type to uint64 without warning. For negative underflow, it crashes. This inconsistent error handling makes it difficult for users to handle edge cases reliably.

5. **Violates data integrity expectations**: Users expect either both overflows to work (with appropriate type conversion) or both to fail with clear errors. The current behavior can lead to subtle bugs where some large integers work and others crash unpredictably.

## Relevant Context

The bug originates in the `ujson_loads` function used by pandas at `/pandas/io/json/_json.py:1392`. Testing confirms that `pandas._libs.json.ujson_loads` exhibits this asymmetric behavior:

- `ujson_loads('[-9223372036854775809]')` raises `ValueError: Value is too small`
- `ujson_loads('[9223372036854775808]')` returns `[9223372036854775808]` successfully

This is particularly problematic in data science contexts where:
- Large integer IDs or timestamps might naturally exceed int64 bounds
- Data from external sources (APIs, databases) may contain arbitrary precision integers
- Users need predictable behavior for data validation and error handling

The JSON specification (RFC 8259, Section 6) explicitly allows implementations to set numeric limits, but implementations should handle limits consistently. The current asymmetric behavior suggests an oversight in the ujson integration rather than an intentional design choice.

## Proposed Fix

The fix requires modifying the ujson integration in pandas to handle underflow consistently with overflow. There are two valid approaches:

**Option 1 (Recommended): Accept both overflows and convert to appropriate types**
```diff
--- a/pandas/io/json/_json.py
+++ b/pandas/io/json/_json.py
@@ -1389,8 +1389,14 @@ class FrameParser(Parser):

         if orient == "columns":
             self.obj = DataFrame(
-                ujson_loads(json, precise_float=self.precise_float), dtype=None
+                self._safe_ujson_loads(json), dtype=None
             )
+
+    def _safe_ujson_loads(self, json_str):
+        try:
+            return ujson_loads(json_str, precise_float=self.precise_float)
+        except ValueError as e:
+            if "Value is too small" in str(e):
+                # Fall back to standard json for values outside int64 range
+                import json as stdlib_json
+                return stdlib_json.loads(json_str)
+            raise
```

**Option 2: Reject both overflows consistently**
```diff
--- a/pandas/io/json/_json.py
+++ b/pandas/io/json/_json.py
@@ -1389,8 +1389,15 @@ class FrameParser(Parser):

         if orient == "columns":
+            data = ujson_loads(json, precise_float=self.precise_float)
+            # Check for uint64 values that indicate overflow
+            if isinstance(data, list):
+                for item in data:
+                    if isinstance(item, dict):
+                        for v in item.values():
+                            if isinstance(v, int) and v > 2**63 - 1:
+                                raise ValueError(f"Value {v} is too large for int64")
             self.obj = DataFrame(
-                ujson_loads(json, precise_float=self.precise_float), dtype=None
+                data, dtype=None
             )
```

Option 1 is recommended as it provides better user experience and maintains backward compatibility for positive overflow cases while fixing the negative underflow crash.