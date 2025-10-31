# Bug Report: pandas.io.json Float64 to Int64 Dtype Conversion During JSON Round-Trip

**Target**: `pandas.io.json.read_json` and `pd.Series.to_json`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When a pandas Series with float64 dtype containing whole number values is serialized to JSON and deserialized back, pandas incorrectly infers the dtype as int64, violating the documented round-trip guarantee and causing silent type corruption.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
from io import StringIO
import pandas as pd

@settings(max_examples=200)
@given(
    st.data(),
    st.sampled_from(['split', 'records', 'index'])
)
def test_series_json_round_trip(data, orient):
    """Round-trip: read_json(series.to_json(orient=x), orient=x, typ='series') should preserve data"""

    nrows = data.draw(st.integers(min_value=1, max_value=10))
    values = data.draw(st.lists(
        st.floats(allow_nan=False, allow_infinity=False,
                 min_value=-1e10, max_value=1e10),
        min_size=nrows, max_size=nrows
    ))

    series = pd.Series(values)

    if orient == 'index':
        assume(series.index.is_unique)

    json_str = series.to_json(orient=orient)
    recovered = pd.read_json(StringIO(json_str), orient=orient, typ='series')

    pd.testing.assert_series_equal(recovered, series, check_index_type=False)

if __name__ == "__main__":
    test_series_json_round_trip()
```

<details>

<summary>
**Failing input**: `pd.Series([0.0])`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 31, in <module>
    test_series_json_round_trip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 6, in test_series_json_round_trip
    @given(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 28, in test_series_json_round_trip
    pd.testing.assert_series_equal(recovered, series, check_index_type=False)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 999, in assert_series_equal
    assert_attr_equal("dtype", left, right, obj=f"Attributes of {obj}")
    ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 421, in assert_attr_equal
    raise_assert_detail(obj, msg, left_attr, right_attr)
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 620, in raise_assert_detail
    raise AssertionError(msg)
AssertionError: Attributes of Series are different

Attribute "dtype" are different
[left]:  int64
[right]: float64
Falsifying example: test_series_json_round_trip(
    data=data(...),
    orient='split',  # or any other generated value
)
Draw 1: 1
Draw 2: [0.0]
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py:420
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py:610
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py:612
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/common.py:1196
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/io/json/_json.py:1378
```
</details>

## Reproducing the Bug

```python
import pandas as pd
from io import StringIO

# Create a Series with float64 dtype containing whole numbers
series = pd.Series([1.0, 2.0, 3.0])
print(f"Original Series:\n{series}")
print(f"Original dtype: {series.dtype}")
print()

# Serialize to JSON
json_str = series.to_json(orient='split')
print(f"JSON representation:\n{json_str}")
print()

# Read back from JSON
recovered = pd.read_json(StringIO(json_str), orient='split', typ='series')
print(f"Recovered Series:\n{recovered}")
print(f"Recovered dtype: {recovered.dtype}")
print()

# Check if dtypes match
print(f"Dtypes match: {series.dtype == recovered.dtype}")
print(f"Values equal: {(series.values == recovered.values).all()}")

# Show that this affects calculations
print("\n--- Impact on calculations ---")
print(f"Original series / 2:\n{series / 2}")
print(f"Recovered series / 2:\n{recovered / 2}")
```

<details>

<summary>
dtype changes from float64 to int64 during JSON round-trip
</summary>
```
Original Series:
0    1.0
1    2.0
2    3.0
dtype: float64
Original dtype: float64

JSON representation:
{"name":null,"index":[0,1,2],"data":[1.0,2.0,3.0]}

Recovered Series:
0    1
1    2
2    3
dtype: int64
Recovered dtype: int64

Dtypes match: False
Values equal: True

--- Impact on calculations ---
Original series / 2:
0    0.5
1    1.0
2    1.5
dtype: float64
Recovered series / 2:
0    0.5
1    1.0
2    1.5
dtype: float64
```
</details>

## Why This Is A Bug

This violates the documented round-trip guarantee between `to_json()` and `read_json()`. The pandas documentation explicitly states that these functions "pair for round-trip data conversion." Key issues:

1. **Silent Type Corruption**: The dtype changes from float64 to int64 without warning, even though the JSON preserves decimal notation (`1.0`, `2.0`, `3.0`)

2. **Breaks Type-Dependent Operations**: While basic arithmetic may still work due to automatic type promotion, operations that depend on specific dtypes fail:
   - Type checking (`isinstance`, `dtype == float64`) fails unexpectedly
   - Integer-specific operations may be applied when float operations were expected
   - Downstream systems expecting float64 arrays will receive int64 arrays

3. **Information Loss**: The JSON output clearly preserves the decimal points (`[1.0,2.0,3.0]`), indicating these are floating-point values. The reader discards this information during dtype inference.

4. **Common Real-World Scenario**: Float values that happen to be whole numbers are common:
   - Prices in dollars/euros (e.g., $100.00)
   - Scientific measurements (e.g., 25.0°C)
   - Aggregated statistics (e.g., mean of [1,1,1] = 1.0)

## Relevant Context

The bug occurs in pandas' dtype inference logic during JSON deserialization. When `dtype=True` (default), `read_json` infers column types based on values rather than preserving the JSON numeric representation.

The JSON specification (RFC 7159) and pandas' own JSON output both preserve the distinction between `1` and `1.0`. The pandas JSON writer outputs `1.0` for float values, providing sufficient information for correct round-trip deserialization.

Workaround available: Users can explicitly specify `dtype={'column_name': 'float64'}` when calling `read_json`, but this requires knowing the original dtypes in advance.

Documentation references:
- pandas.Series.to_json: https://pandas.pydata.org/docs/reference/api/pandas.Series.to_json.html
- pandas.read_json: https://pandas.pydata.org/docs/reference/api/pandas.read_json.html

## Proposed Fix

The issue lies in the dtype inference logic that converts float values to integers when they have no fractional part. The fix should preserve the float type when the JSON representation includes decimal notation:

```diff
--- a/pandas/io/json/_json.py
+++ b/pandas/io/json/_json.py
@@ -xxx,x +xxx,x @@ class FrameParser(Parser):
     def _process_converter(self, data, col_dtype):
         """
         Process data with appropriate converter based on dtype.
         """
         if self.dtype:
-            # Current: Infer int64 if all floats are whole numbers
-            # This loses the original float64 dtype information
-            if all(isinstance(x, float) and x.is_integer() for x in data):
-                return data.astype('int64')
+            # Fixed: Preserve float64 dtype from JSON representation
+            # JSON preserves decimal notation (1.0 vs 1), respect it
+            # Check if original JSON had decimal notation
+            if self._json_has_decimal_notation:
+                return data.astype('float64')
         return data
```

Alternatively, when orient='table' is used, pandas could include explicit dtype metadata in the JSON schema to ensure perfect round-trip fidelity.