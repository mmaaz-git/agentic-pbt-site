# Bug Report: pandas.io.sas.sas7bdat._parse_datetime OverflowError on Large Values

**Target**: `pandas.io.sas.sas7bdat._parse_datetime`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `_parse_datetime` function crashes with an `OverflowError` when processing float values exceeding Python's `timedelta` limits, instead of handling them gracefully like it does for NaN values.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.io.sas.sas7bdat import _parse_datetime

@given(st.floats(allow_nan=False, allow_infinity=False, min_value=1e13, max_value=1e16))
def test_parse_datetime_handles_large_values(x):
    try:
        result = _parse_datetime(x, 's')
    except OverflowError:
        raise AssertionError(f"_parse_datetime crashed with OverflowError for value {x}")

# Run the test
if __name__ == "__main__":
    test_parse_datetime_handles_large_values()
```

<details>

<summary>
**Failing input**: `10000000000000.0` (1e13)
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 7, in test_parse_datetime_handles_large_values
    result = _parse_datetime(x, 's')
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/sas/sas7bdat.py", line 72, in _parse_datetime
    return datetime(1960, 1, 1) + timedelta(seconds=sas_datetime)
           ~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
OverflowError: date value out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 13, in <module>
    test_parse_datetime_handles_large_values()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 5, in test_parse_datetime_handles_large_values
    def test_parse_datetime_handles_large_values(x):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 9, in test_parse_datetime_handles_large_values
    raise AssertionError(f"_parse_datetime crashed with OverflowError for value {x}")
AssertionError: _parse_datetime crashed with OverflowError for value 10000000000000.0
Falsifying example: test_parse_datetime_handles_large_values(
    x=10000000000000.0,
)
```
</details>

## Reproducing the Bug

```python
from pandas.io.sas.sas7bdat import _parse_datetime

# Test with large value that causes OverflowError
large_value = 1e15

try:
    print(f"Testing _parse_datetime with value {large_value} (seconds)")
    result = _parse_datetime(large_value, 's')
    print(f"Result: {result}")
except OverflowError as e:
    print(f"OverflowError: {e}")

print()

try:
    print(f"Testing _parse_datetime with value {large_value} (days)")
    result = _parse_datetime(large_value, 'd')
    print(f"Result: {result}")
except OverflowError as e:
    print(f"OverflowError: {e}")

print()

# Test with NaN to show inconsistent handling
import numpy as np
print("Testing _parse_datetime with NaN value")
result = _parse_datetime(np.nan, 's')
print(f"NaN handling result: {result}")

print()

# Test with normal values that work
normal_value = 100000
print(f"Testing _parse_datetime with normal value {normal_value} (seconds)")
result = _parse_datetime(normal_value, 's')
print(f"Result: {result}")
```

<details>

<summary>
OverflowError when parsing large datetime values
</summary>
```
Testing _parse_datetime with value 1000000000000000.0 (seconds)
OverflowError: Python int too large to convert to C int

Testing _parse_datetime with value 1000000000000000.0 (days)
OverflowError: Python int too large to convert to C int

Testing _parse_datetime with NaN value
NaN handling result: NaT

Testing _parse_datetime with normal value 100000 (seconds)
Result: 1960-01-02 03:46:40
```
</details>

## Why This Is A Bug

This violates expected behavior for several critical reasons:

1. **Inconsistent error handling**: The function gracefully handles NaN values by returning `pd.NaT`, but crashes with `OverflowError` for out-of-range values. This inconsistency creates unpredictable behavior.

2. **Real-world impact**: Corrupt or malformed SAS files are common in data processing pipelines. When `pandas.read_sas()` encounters such files with invalid datetime values, it crashes the entire read operation rather than handling the bad data gracefully.

3. **Undocumented limits**: The function provides no documentation about valid input ranges. Python's `timedelta` has a maximum of approximately 9.2 billion seconds (about 292 years), but values like 1e13 seconds (317,097 years) cause overflow when added to the 1960 epoch.

4. **Poor error messaging**: The generic "date value out of range" or "Python int too large to convert to C int" errors provide no context about the SAS file parsing failure, making debugging difficult for users.

5. **Violates pandas conventions**: Other pandas datetime functions handle invalid values more gracefully, typically converting them to `NaT` rather than crashing.

## Relevant Context

The `_parse_datetime` function is located at `/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/sas/sas7bdat.py:67-78`. It's an internal function (indicated by the leading underscore) used by the public `pandas.read_sas()` API when processing SAS7BDAT format files.

SAS uses January 1, 1960 as its datetime epoch. The function converts SAS datetime values (seconds or days since 1960-01-01) to Python datetime objects. However, Python's `timedelta` has inherent limits that the function doesn't account for:
- Maximum seconds: ~9.2 billion (sys.maxsize // 10**9)
- Maximum days: ~106,751 (sys.maxsize // (24*3600*10**9))

Values exceeding these limits when added to datetime(1960, 1, 1) cause the OverflowError.

The pandas codebase already has a more robust datetime conversion function `_convert_datetimes` (lines 81-108 in the same file) that uses vectorized operations and handles edge cases better. However, `_parse_datetime` is still used in some code paths.

## Proposed Fix

```diff
--- a/pandas/io/sas/sas7bdat.py
+++ b/pandas/io/sas/sas7bdat.py
@@ -67,11 +67,17 @@ def _parse_datetime(sas_datetime: float, unit: str):
 def _parse_datetime(sas_datetime: float, unit: str):
     if isna(sas_datetime):
         return pd.NaT

-    if unit == "s":
-        return datetime(1960, 1, 1) + timedelta(seconds=sas_datetime)
-
-    elif unit == "d":
-        return datetime(1960, 1, 1) + timedelta(days=sas_datetime)
+    try:
+        if unit == "s":
+            return datetime(1960, 1, 1) + timedelta(seconds=sas_datetime)
+
+        elif unit == "d":
+            return datetime(1960, 1, 1) + timedelta(days=sas_datetime)
+    except (OverflowError, ValueError):
+        # Handle out-of-range values the same as NaN
+        return pd.NaT

     else:
         raise ValueError("unit must be 'd' or 's'")
```