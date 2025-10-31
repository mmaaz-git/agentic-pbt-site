# Bug Report: pandas.io.json Large Float Becomes Infinity During JSON Round-Trip

**Target**: `pandas.DataFrame.to_json()` and `pandas.read_json()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When serializing and deserializing DataFrames containing large but valid finite floats near `sys.float_info.max` through pandas' JSON methods, the values are silently corrupted and become infinity due to precision truncation in the underlying ujson library.

## Property-Based Test

```python
from io import StringIO
import pandas as pd
from hypothesis import given, settings, strategies as st, assume, example


@given(
    st.lists(
        st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=5),
        min_size=1,
        max_size=10
    )
)
@example([[1.7976931345e+308]])  # Specific failing case
@settings(max_examples=500)
def test_dataframe_roundtrip_columns(data):
    num_cols = len(data[0])
    assume(all(len(row) == num_cols for row in data))

    df = pd.DataFrame(data)
    assume(df.columns.is_unique)

    json_str = df.to_json(orient='columns')
    df_recovered = pd.read_json(StringIO(json_str), orient='columns')

    pd.testing.assert_frame_equal(df, df_recovered, check_dtype=False)


if __name__ == "__main__":
    # Run the test
    print("Running Hypothesis test for pandas JSON round-trip...")
    print("=" * 60)

    try:
        test_dataframe_roundtrip_columns()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed with AssertionError:")
        print(str(e))
        print("\nFalsifying example: test_dataframe_roundtrip_columns(")
        print("    data=[[1.7976931345e+308]],")
        print(")")
    except Exception as e:
        print(f"Test failed with unexpected error:")
        print(str(e))
```

<details>

<summary>
**Failing input**: `data=[[1.7976931345e+308]]`
</summary>
```
Running Hypothesis test for pandas JSON round-trip...
============================================================
Test failed with AssertionError:
DataFrame.iloc[:, 0] (column name="0") are different

DataFrame.iloc[:, 0] (column name="0") values are different (100.0 %)
[index]: [0]
[left]:  [1.7976931345e+308]
[right]: [inf]
At positional index 0, first diff: 1.7976931345e+308 != inf

Falsifying example: test_dataframe_roundtrip_columns(
    data=[[1.7976931345e+308]],
)
```
</details>

## Reproducing the Bug

```python
from io import StringIO
import pandas as pd
import sys

print("=" * 60)
print("Demonstrating pandas JSON Large Float Becomes Infinity Bug")
print("=" * 60)

# Test with the specific value that triggers the bug
test_value = 1.7976931345e+308

print(f"\n1. Input validation:")
print(f"   Test value: {test_value}")
print(f"   sys.float_info.max: {sys.float_info.max}")
print(f"   Is test value finite: {test_value != float('inf') and test_value != float('-inf')}")
print(f"   Is test value < max: {test_value < sys.float_info.max}")

# Create DataFrame with the problematic value
df = pd.DataFrame([[test_value]])
print(f"\n2. Original DataFrame:")
print(f"   Value at [0,0]: {df.iloc[0, 0]}")
print(f"   Type: {type(df.iloc[0, 0])}")
print(f"   Is finite: {df.iloc[0, 0] != float('inf')}")

# Serialize to JSON
json_str = df.to_json(orient='columns')
print(f"\n3. JSON serialization (default double_precision=10):")
print(f"   JSON string: {json_str}")

# Deserialize from JSON
df_recovered = pd.read_json(StringIO(json_str), orient='columns')
print(f"\n4. Recovered DataFrame:")
print(f"   Value at [0,0]: {df_recovered.iloc[0, 0]}")
print(f"   Type: {type(df_recovered.iloc[0, 0])}")
print(f"   Is infinite: {df_recovered.iloc[0, 0] == float('inf')}")

# Show the problem
print(f"\n5. Data corruption detected:")
print(f"   Original == Recovered: {df.iloc[0, 0] == df_recovered.iloc[0, 0]}")
print(f"   Original value: {df.iloc[0, 0]}")
print(f"   Recovered value: {df_recovered.iloc[0, 0]}")
print(f"   DATA CORRUPTED: finite value became infinity!")

# Test with higher precision
print(f"\n6. Testing with double_precision=15:")
json_str_15 = df.to_json(orient='columns', double_precision=15)
print(f"   JSON string: {json_str_15}")
df_recovered_15 = pd.read_json(StringIO(json_str_15), orient='columns')
print(f"   Recovered value: {df_recovered_15.iloc[0, 0]}")
print(f"   Original == Recovered: {df.iloc[0, 0] == df_recovered_15.iloc[0, 0]}")

# Compare with standard library
import json
print(f"\n7. Standard library json for comparison:")
data_dict = {"0": {"0": test_value}}
std_json = json.dumps(data_dict)
print(f"   JSON string: {std_json}")
std_recovered = json.loads(std_json)
print(f"   Recovered value: {std_recovered['0']['0']}")
print(f"   Original == Recovered: {test_value == std_recovered['0']['0']}")
```

<details>

<summary>
Output demonstrating data corruption
</summary>
```
============================================================
Demonstrating pandas JSON Large Float Becomes Infinity Bug
============================================================

1. Input validation:
   Test value: 1.7976931345e+308
   sys.float_info.max: 1.7976931348623157e+308
   Is test value finite: True
   Is test value < max: True

2. Original DataFrame:
   Value at [0,0]: 1.7976931345e+308
   Type: <class 'numpy.float64'>
   Is finite: True

3. JSON serialization (default double_precision=10):
   JSON string: {"0":{"0":1.797693135e+308}}

4. Recovered DataFrame:
   Value at [0,0]: inf
   Type: <class 'numpy.float64'>
   Is infinite: True

5. Data corruption detected:
   Original == Recovered: False
   Original value: 1.7976931345e+308
   Recovered value: inf
   DATA CORRUPTED: finite value became infinity!

6. Testing with double_precision=15:
   JSON string: {"0":{"0":1.7976931345e+308}}
   Recovered value: 1.7976931345e+308
   Original == Recovered: True

7. Standard library json for comparison:
   JSON string: {"0": {"0": 1.7976931345e+308}}
   Recovered value: 1.7976931345e+308
   Original == Recovered: True
```
</details>

## Why This Is A Bug

This violates expected behavior in multiple critical ways:

1. **Silent Data Corruption**: A valid, finite float value (`1.7976931345e+308`) is transformed into infinity without any warning or error. The value is legitimately less than `sys.float_info.max` (1.7976931348623157e+308) and should remain finite.

2. **Violates JSON Round-Trip Invariant**: The fundamental expectation of JSON serialization is that valid data can be round-tripped without corruption. pandas' `to_json`/`read_json` breaks this contract for certain valid float values.

3. **Inconsistent with Standard Library**: Python's built-in `json` module correctly handles the same value, proving that proper handling is both possible and expected. The standard library preserves the value as `1.7976931345e+308` through the round-trip.

4. **Precision Truncation Cascade**: The bug occurs because:
   - `to_json()` with default `double_precision=10` truncates `1.7976931345e+308` to `1.797693135e+308`
   - This truncated representation, when parsed by Python or ujson, exceeds the maximum representable finite float
   - The parser then interprets it as infinity instead of raising an error

5. **Documentation Gap**: While pandas documents that `double_precision` has a maximum of 15 and defaults to 10, it does not warn users that this can cause valid finite values to become infinity. This is not merely a precision loss but a qualitative change from finite to infinite.

## Relevant Context

- **Affected versions**: All pandas versions using ujson with default `double_precision=10`
- **Related GitHub Issue**: pandas #38437 acknowledges precision limitations but doesn't specifically document the infinity corruption issue
- **IEEE 754 Context**: The value `1.7976931345e+308` requires more than 10 digits of precision to remain below the finite float limit when round-tripped
- **Use cases affected**: Scientific computing, financial modeling, physics simulations, or any domain working with large magnitude values
- **Workaround available**: Setting `double_precision=15` in `to_json()` prevents the corruption but isn't discoverable without encountering the bug first

Key documentation references:
- [pandas.DataFrame.to_json](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_json.html) - See `double_precision` parameter
- [pandas.read_json](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_json.html) - See `precise_float` parameter

## Proposed Fix

The issue requires fixing at multiple levels. Here's a minimal defensive fix for pandas:

```diff
--- a/pandas/io/json/_json.py
+++ b/pandas/io/json/_json.py
@@ -130,6 +130,16 @@ def to_json(
         )

     if double_precision < 0 or double_precision > 15:
         raise ValueError("double_precision out of range [0,15]")
+
+    # Warn users about potential infinity corruption for large floats with low precision
+    if double_precision < 15:
+        import warnings
+        warnings.warn(
+            f"Using double_precision={double_precision} may cause large float values "
+            f"near sys.float_info.max to become infinity after deserialization. "
+            f"Consider using double_precision=15 for applications requiring the full "
+            f"float range.",
+            category=RuntimeWarning
+        )

     # Rest of the function...
```

A more comprehensive fix would involve:
1. Detecting when serialized values would parse as infinity and either raising an error or automatically increasing precision
2. Contributing a fix to the upstream ujson library to handle edge cases correctly
3. Adding explicit test cases for values near `sys.float_info.max` to prevent regression
4. Documenting the limitation clearly in the `to_json` and `read_json` docstrings