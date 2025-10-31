# Bug Report: pandas.io.parsers CSV Large Integer Silent Type Corruption

**Target**: `pandas.io.parsers.read_csv` and `pandas.DataFrame.to_csv`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When reading CSV files containing integer values beyond the int64 range (-2^63 to 2^63-1), pandas silently converts these integers to strings instead of preserving them as Python int objects, causing data corruption and breaking arithmetic operations.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
import pandas as pd
import io
from pandas.testing import assert_frame_equal

@given(
    st.lists(st.integers(), min_size=1, max_size=50),
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10), min_size=1, max_size=50)
)
@settings(max_examples=100)
def test_csv_round_trip_basic(int_data, float_data):
    """Round-trip property: read_csv(to_csv(df)) should equal df (basic types)"""
    assume(len(int_data) == len(float_data))

    df = pd.DataFrame({'int_col': int_data, 'float_col': float_data})

    csv_str = df.to_csv(index=False)
    result = pd.read_csv(io.StringIO(csv_str))

    assert len(result) == len(df)
    assert list(result.columns) == list(df.columns)
    assert_frame_equal(result, df)

if __name__ == "__main__":
    test_csv_round_trip_basic()
```

<details>

<summary>
**Failing input**: `int_data=[0, 0, -9223372036854775809], float_data=[0.0, 0.0, 0.0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 25, in <module>
    test_csv_round_trip_basic()
    ~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 7, in test_csv_round_trip_basic
    st.lists(st.integers(), min_size=1, max_size=50),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 22, in test_csv_round_trip_basic
    assert_frame_equal(result, df)
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^
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
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 1091, in assert_series_equal
    _testing.assert_almost_equal(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        left._values,
        ^^^^^^^^^^^^^
    ...<5 lines>...
        index_values=left.index,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "pandas/_libs/testing.pyx", line 55, in pandas._libs.testing.assert_almost_equal
  File "pandas/_libs/testing.pyx", line 173, in pandas._libs.testing.assert_almost_equal
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 620, in raise_assert_detail
    raise AssertionError(msg)
AssertionError: DataFrame.iloc[:, 0] (column name="int_col") are different

DataFrame.iloc[:, 0] (column name="int_col") values are different (100.0 %)
[index]: [0, 1, 2]
[left]:  [0, 0, -9223372036854775809]
[right]: [0, 0, -9223372036854775809]
At positional index 0, first diff: 0 != 0
Falsifying example: test_csv_round_trip_basic(
    int_data=[0, 0, -9_223_372_036_854_775_809],
    float_data=[0.0, 0.0, 0.0],
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import io

# Test with integer beyond int64 range
int_value = -9223372036854775809  # Less than min int64 (-2^63)

# Create DataFrame with large integer
df = pd.DataFrame({'int_col': [int_value]})
print(f"Original value: {df['int_col'].iloc[0]}")
print(f"Original type: {type(df['int_col'].iloc[0]).__name__}")
print(f"Original dtype: {df['int_col'].dtype}")

# Round-trip through CSV
csv_str = df.to_csv(index=False)
print(f"\nCSV content:\n{csv_str}")

# Read back from CSV
result = pd.read_csv(io.StringIO(csv_str))
print(f"After round-trip value: {result['int_col'].iloc[0]}")
print(f"After round-trip type: {type(result['int_col'].iloc[0]).__name__}")
print(f"After round-trip dtype: {result['int_col'].dtype}")

# Check equality
print(f"\nValues equal: {df['int_col'].iloc[0] == result['int_col'].iloc[0]}")
print(f"Types match: {type(df['int_col'].iloc[0]) == type(result['int_col'].iloc[0])}")

# Demonstrate the problem - arithmetic operations fail
try:
    original_arithmetic = df['int_col'].iloc[0] + 1
    print(f"\nOriginal + 1 = {original_arithmetic}")
except Exception as e:
    print(f"\nOriginal + 1 failed: {e}")

try:
    result_arithmetic = result['int_col'].iloc[0] + 1
    print(f"After round-trip + 1 = {result_arithmetic}")
except Exception as e:
    print(f"After round-trip + 1 failed: {e}")
```

<details>

<summary>
Silent type corruption: int becomes str, breaking arithmetic operations
</summary>
```
Original value: -9223372036854775809
Original type: int
Original dtype: object

CSV content:
int_col
-9223372036854775809

After round-trip value: -9223372036854775809
After round-trip type: str
After round-trip dtype: object

Values equal: False
Types match: False

Original + 1 = -9223372036854775808
After round-trip + 1 failed: can only concatenate str (not "int") to str
```
</details>

## Why This Is A Bug

This behavior violates expected data preservation semantics and causes silent data corruption:

1. **Silent Type Corruption**: The integer value is silently converted from `int` to `str` without any warning or error, violating the principle of least surprise.

2. **Data Integrity Violation**: pandas correctly stores the large integer as a Python `int` object in an `object` dtype column when creating the DataFrame. The same data structure (object dtype with Python int) should be restored when reading from CSV, but instead it stores a string.

3. **Arithmetic Operations Break**: After the round-trip, mathematical operations fail with `TypeError: can only concatenate str (not "int") to str`, making the data unusable for numerical computation.

4. **Inconsistent Behavior**: pandas demonstrates it CAN handle large integers (it stores them correctly in the original DataFrame), but fails to restore them properly from CSV. The write operation correctly outputs the numeric value, but the read operation incorrectly interprets it as a string.

5. **No Documentation Warning**: The pandas documentation for `read_csv` and `to_csv` does not warn users that integers beyond int64 range will be converted to strings, leaving users unaware of this data corruption risk.

## Relevant Context

This bug affects pandas version 2.3.2 and likely earlier versions. The issue occurs in the CSV parser's type inference logic, which appears to fall back to string dtype when encountering numeric values that overflow int64, rather than attempting to parse them as Python int objects.

The bug is particularly problematic for:
- Financial applications dealing with large monetary values
- Systems using large integer IDs or timestamps beyond Unix epoch range
- Scientific computing with large integer counts
- Any application requiring precise integer arithmetic on large values

Related GitHub issues:
- Issue #52505: "BUG: incorrect reading of CSV containing large integers" - confirms this is a known problem area
- The pandas team has acknowledged similar issues as bugs requiring fixes

## Proposed Fix

The fix requires modifying the CSV parser's type inference to attempt parsing overflow integers as Python int objects before falling back to string. The logic should be:

1. Try to parse as int64
2. If overflow occurs, try to parse as Python int object
3. Only fall back to string if parsing as int fails

Here's a conceptual patch for the type inference logic:

```diff
# In pandas/_libs/parsers.pyx or equivalent type inference code
def infer_type_for_numeric_string(value_str):
    try:
        # Try int64 first for performance
        return np.int64(value_str)
    except OverflowError:
-       # Currently: falls back to string immediately
-       return value_str
+       # New: Try Python int for values outside int64 range
+       try:
+           return int(value_str)
+       except ValueError:
+           # Only fall back to string if it's not a valid integer
+           return value_str
    except ValueError:
        # Not a valid integer at all
        return value_str
```

This ensures that large integers remain as numeric Python int objects in object dtype columns, preserving both the numeric type and arithmetic capabilities after CSV round-trip operations.