# Bug Report: pandas.io.parsers.read_csv Large Integer Silent Type Conversion

**Target**: `pandas.io.parsers.read_csv`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `read_csv` function silently converts Python integers that exceed int64/uint64 bounds to strings, causing type inconsistency in CSV round-trip operations.

## Property-Based Test

```python
import io
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings


@given(
    overflow_amount=st.integers(min_value=1, max_value=10**15)
)
@settings(max_examples=100)
def test_large_int_round_trip_preserves_type(overflow_amount):
    """
    Property: CSV round-trip should preserve numeric types, including integers
    that don't fit in int64.

    Evidence: DataFrames can store Python integers of arbitrary size (as object dtype),
    and CSV round-trip should preserve this data type, not convert to string.
    """
    int64_min = np.iinfo(np.int64).min
    overflow_value = int64_min - overflow_amount

    df = pd.DataFrame({'x': [overflow_value]})

    assert df['x'].dtype == object
    assert isinstance(df['x'].iloc[0], int)

    csv_string = df.to_csv(index=False)
    result = pd.read_csv(io.StringIO(csv_string))

    assert isinstance(result['x'].iloc[0], int), \
        f"Expected int, got {type(result['x'].iloc[0]).__name__}"
    assert result['x'].iloc[0] == overflow_value


if __name__ == "__main__":
    test_large_int_round_trip_preserves_type()
```

<details>

<summary>
**Failing input**: `overflow_amount=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo.py", line 36, in <module>
    test_large_int_round_trip_preserves_type()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo.py", line 8, in test_large_int_round_trip_preserves_type
    overflow_amount=st.integers(min_value=1, max_value=10**15)
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo.py", line 30, in test_large_int_round_trip_preserves_type
    assert isinstance(result['x'].iloc[0], int), \
           ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected int, got str
Falsifying example: test_large_int_round_trip_preserves_type(
    overflow_amount=1,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import io
import pandas as pd
import numpy as np

# Test with a value just outside int64 range
int64_min = np.iinfo(np.int64).min
overflow_value = int64_min - 1

print(f"Testing with value: {overflow_value}")
print(f"int64_min: {int64_min}")
print(f"overflow_value: {overflow_value}")
print()

# Create DataFrame with large integer
df = pd.DataFrame({'x': [overflow_value]})

print(f"Original DataFrame:")
print(df)
print(f"Original type: {type(df['x'].iloc[0])}")
print(f"Original dtype: {df['x'].dtype}")
print(f"Original value: {df['x'].iloc[0]}")
print()

# Convert to CSV and read back
csv_string = df.to_csv(index=False)
print(f"CSV string representation:")
print(csv_string)

result = pd.read_csv(io.StringIO(csv_string))

print(f"After CSV round-trip:")
print(result)
print(f"After CSV round-trip type: {type(result['x'].iloc[0])}")
print(f"After CSV round-trip dtype: {result['x'].dtype}")
print(f"After CSV round-trip value: {result['x'].iloc[0]}")
print()

# Verify the bug
print("Assertion results:")
print(f"Original is int: {isinstance(df['x'].iloc[0], int)}")
print(f"After round-trip is str: {isinstance(result['x'].iloc[0], str)}")
print(f"Values are equal (as strings): {str(df['x'].iloc[0]) == str(result['x'].iloc[0])}")

# This demonstrates the bug: numeric data became string data
assert isinstance(df['x'].iloc[0], int), "Original value should be int"
assert isinstance(result['x'].iloc[0], str), "Bug: After round-trip, value is string instead of int"
```

<details>

<summary>
Silent type conversion from int to str for value -9223372036854775809
</summary>
```
Testing with value: -9223372036854775809
int64_min: -9223372036854775808
overflow_value: -9223372036854775809

Original DataFrame:
                      x
0  -9223372036854775809
Original type: <class 'int'>
Original dtype: object
Original value: -9223372036854775809

CSV string representation:
x
-9223372036854775809

After CSV round-trip:
                      x
0  -9223372036854775809
After CSV round-trip type: <class 'str'>
After CSV round-trip dtype: object
After CSV round-trip value: -9223372036854775809

Assertion results:
Original is int: True
After round-trip is str: True
Values are equal (as strings): True
```
</details>

## Why This Is A Bug

This behavior violates fundamental data integrity expectations for the following specific reasons:

1. **Inconsistent Type Inference**: The parser correctly handles integers within int64 bounds (−9,223,372,036,854,775,808 to 9,223,372,036,854,775,807) and uint64 bounds (0 to 18,446,744,073,709,551,615), but silently converts larger values to strings. This creates an undocumented discontinuity in behavior at specific numeric boundaries.

2. **Silent Data Corruption**: The conversion from numeric to string type happens without warning or error, potentially breaking downstream numeric operations. Code that expects `df['column'].sum()` to work will fail with a TypeError when the column contains string values.

3. **Contradicts DataFrame Capabilities**: Pandas DataFrames explicitly support arbitrary-precision Python integers via object dtype. The CSV writer correctly outputs these values as numeric text, but the reader fails to parse them back as integers.

4. **Violates CSV Round-Trip Invariant**: Users reasonably expect that `pd.read_csv(df.to_csv())` preserves data types for supported types. Since DataFrames support arbitrary-precision integers, the CSV operations should maintain this support.

5. **Undocumented Behavior**: The pandas documentation for `read_csv` does not mention that integers outside int64/uint64 bounds will be converted to strings. The documentation states the parser "attempts to coerce columns to numeric types when possible" without defining the failure mode.

## Relevant Context

Testing reveals the exact boundaries of this bug:
- Values within int64 range: correctly parsed as int64
- Values within uint64 range but outside int64: correctly parsed as uint64
- Values outside both ranges: incorrectly parsed as string

A working solution already exists within pandas - the `converters` parameter:
```python
result = pd.read_csv(io.StringIO(csv_string), converters={'x': int})
```
This successfully parses large integers, proving the capability exists but isn't used by default.

Related GitHub Issue #52505 acknowledges this as a bug requiring fixes. While partially addressed for the pyarrow engine, the default C engine still exhibits this behavior.

Documentation: `/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/parsers/base_parser.py:680`
Parser implementation: `/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/parsers/c_parser_wrapper.py`

## Proposed Fix

The fix requires modifying the type inference logic in the CSV parser to attempt Python `int()` conversion when numeric strings exceed int64/uint64 bounds. The change would be in the parser's `_infer_types` method to add a fallback path for large integers:

```diff
--- a/pandas/io/parsers/base_parser.py
+++ b/pandas/io/parsers/base_parser.py
@@ -720,6 +720,15 @@ class ParserBase:
                 result, na_count = self._infer_string_dtype(
                     values, na_values, no_dtype_specified
                 )
+                # Check if string values might be large integers
+                if result.dtype == object and na_count == 0:
+                    try:
+                        # Try converting to Python int for values outside int64/uint64
+                        test_vals = [int(x) for x in result if x not in na_values]
+                        if all(isinstance(v, int) for v in test_vals):
+                            result = np.array(test_vals, dtype=object)
+                    except (ValueError, TypeError):
+                        pass  # Keep as string if conversion fails

         else:
             result = values
```

Alternatively, enhance the numeric conversion logic to detect overflow and use Python integers:

```diff
--- a/pandas/_libs/parsers.pyx
+++ b/pandas/_libs/parsers.pyx
@@ -1234,8 +1234,14 @@ cdef class TextReader:
                     if not overflow:
                         col[i] = val
                     else:
-                        # Overflow, treat as string
-                        col = self._convert_to_string_array(col, i)
+                        # Try Python int for overflow values
+                        try:
+                            py_val = int(word)
+                            col = self._convert_to_object_array(col, i)
+                            col[i] = py_val
+                        except ValueError:
+                            # Fall back to string
+                            col = self._convert_to_string_array(col, i)
```