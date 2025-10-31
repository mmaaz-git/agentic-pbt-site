# Bug Report: pandas.core.interchange date32 Conversion Corrupts Pre-Epoch Dates

**Target**: `pandas.core.interchange.from_dataframe.parse_datetime_format_str`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `parse_datetime_format_str()` function incorrectly converts date32[day] values to uint64 before multiplication, causing dates before the Unix epoch (1970-01-01) to wrap around to incorrect future dates.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
from pandas.core.interchange.from_dataframe import parse_datetime_format_str

@given(st.integers(min_value=-10000, max_value=10000))
@settings(max_examples=500)
def test_date32_preserves_sign(days_from_epoch):
    data = np.array([days_from_epoch], dtype=np.int32)
    result = parse_datetime_format_str("tdD", data)

    expected_seconds = days_from_epoch * 86400
    result_seconds = result.astype('int64').view('int64')

    assert result_seconds[0] == expected_seconds, \
        f"Date conversion failed: {days_from_epoch} days -> {result_seconds[0]} vs {expected_seconds} seconds"
```

**Failing input**: Any negative value, e.g., `days_from_epoch = -1` (1969-12-31)

## Reproducing the Bug

```python
import numpy as np
from pandas.core.interchange.from_dataframe import parse_datetime_format_str

data = np.array([-1, -5, -10], dtype=np.int32)

print(f"Input: {data} days from epoch")
print("Expected dates: 1969-12-31, 1969-12-27, 1969-12-22")

result = parse_datetime_format_str("tdD", data)
print(f"\nActual result: {result}")
```

**Expected output:**
```
Input: [-1 -5 -10] days from epoch
Expected dates: 1969-12-31, 1969-12-27, 1969-12-22

Actual result: ['1969-12-31' '1969-12-27' '1969-12-22']
```

**Actual output:**
```
Input: [-1 -5 -10] days from epoch
Expected dates: 1969-12-31, 1969-12-27, 1969-12-22

Actual result: ['2555-07-03T21:56:40' '2555-06-30T08:53:20' '2555-06-25T19:50:00']
```

## Why This Is A Bug

**Root cause**: In `parse_datetime_format_str()` (from_dataframe.py:383-385):

```python
if unit == "D":
    # NumPy doesn't support DAY unit, so converting days to seconds
    # (converting to uint64 to avoid overflow)
    data = (data.astype(np.uint64) * (24 * 60 * 60)).astype("datetime64[s]")
```

**The problem:**

1. Input: `data = np.array([-1], dtype=np.int32)` (one day before epoch)
2. **Line 385**: `data.astype(np.uint64)` converts -1 to 18446744073709551615 (2^64 - 1)
3. Multiply by 86400 (seconds per day): results in a massive number
4. Convert to datetime64[s]: produces a date in year 2555 instead of 1969!

**Impact**:
- **Data corruption**: All dates before 1970-01-01 are converted to far-future dates
- **Silent failure**: No error is raised, data is silently corrupted
- **Violates interchange protocol**: date32 in Arrow explicitly supports negative values for pre-epoch dates

**The comment is wrong**: The comment says "converting to uint64 to avoid overflow", but:
- There is no overflow risk with int64 for date32 values
- int32 days ×  86400 fits comfortably in int64 range
- date32 can represent dates from year ~-5.8M to ~+5.8M, well within int64 capacity

## Fix

Use int64 instead of uint64:

```diff
--- a/pandas/core/interchange/from_dataframe.py
+++ b/pandas/core/interchange/from_dataframe.py
@@ -381,8 +381,7 @@ def parse_datetime_format_str(format_str, data) -> pd.Series | np.ndarray:
         unit = date_meta.group(1)
         if unit == "D":
             # NumPy doesn't support DAY unit, so converting days to seconds
-            # (converting to uint64 to avoid overflow)
-            data = (data.astype(np.uint64) * (24 * 60 * 60)).astype("datetime64[s]")
+            data = (data.astype(np.int64) * (24 * 60 * 60)).astype("datetime64[s]")
         elif unit == "m":
             data = data.astype("datetime64[ms]")
         else:
```

This correctly preserves the sign of negative date values, allowing dates before 1970 to be represented accurately.