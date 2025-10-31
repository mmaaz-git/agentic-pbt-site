# Bug Report: pandas.api.interchange Integer Overflow in Day-to-Second Conversion

**Target**: `pandas.core.interchange.from_dataframe.parse_datetime_format_str`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `parse_datetime_format_str` function silently overflows when converting very large day values to seconds, producing negative datetime values instead of raising an error or handling the overflow correctly.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pytest
import numpy as np
from pandas.core.interchange.from_dataframe import parse_datetime_format_str


@given(st.integers(min_value=2**62, max_value=2**63-1))
@settings(max_examples=100)
def test_parse_datetime_days_overflow(days):
    format_str = "tdD"
    data = np.array([days], dtype=np.int64)

    result = parse_datetime_format_str(format_str, data)

    expected_seconds = np.uint64(days) * np.uint64(24 * 60 * 60)

    if expected_seconds > 2**63 - 1:
        result_as_int = result.view('int64')[0]
        assert result_as_int >= 0 or days < 0, \
            f"Silent overflow: positive days={days} produced negative result={result_as_int}"
```

**Failing input**: `days=4_735_838_584_154_958_556` (or any value > 106,751,991,167,300)

## Reproducing the Bug

```python
import numpy as np
from pandas.core.interchange.from_dataframe import parse_datetime_format_str

days = 4_735_838_584_154_958_556

format_str = "tdD"
data = np.array([days], dtype=np.int64)

result = parse_datetime_format_str(format_str, data)

print(f"Input days: {days}")
print(f"Result: {result}")

result_as_int = result.view('int64')[0]
print(f"Result as int64: {result_as_int}")
print(f"Is negative? {result_as_int < 0}")
```

Output:
```
Input days: 4735838584154958556
Result: ['-292277022657-01-28T03:24:48']
Result as int64: -9223372036854707712
Is negative? True
```

## Why This Is A Bug

1. **Silent data corruption**: The function produces incorrect negative datetime values for positive input days without raising an error
2. **Comment misleading**: Line 384 has a comment "(converting to uint64 to avoid overflow)" but the conversion to uint64 doesn't prevent overflow during multiplication
3. **Violates interchange protocol expectations**: When data cannot be represented, the protocol should raise an error rather than silently produce wrong values

The overflow occurs at line 385 when multiplying uint64 by the seconds-per-day constant:
```python
data = (data.astype(np.uint64) * (24 * 60 * 60)).astype('datetime64[s]')
```

For day values > 106,751,991,167,300 (approximately 292 billion years), the multiplication overflows, wraps around, and produces negative datetime values.

## Fix

Add validation to check for overflow before performing the multiplication:

```diff
--- a/pandas/core/interchange/from_dataframe.py
+++ b/pandas/core/interchange/from_dataframe.py
@@ -380,8 +380,12 @@ def parse_datetime_format_str(format_str, data) -> pd.Series | np.ndarray:
     if date_meta:
         unit = date_meta.group(1)
         if unit == "D":
+            # Check for overflow: max valid days for datetime64[s] is (2^63 - 1) / 86400
+            max_days = (2**63 - 1) // (24 * 60 * 60)
+            if np.any(np.abs(data) > max_days):
+                raise OverflowError(f"Day values must be in range [{-max_days}, {max_days}] to convert to datetime64[s]")
             # NumPy doesn't support DAY unit, so converting days to seconds
-            # (converting to uint64 to avoid overflow)
+            # Converting to uint64 to avoid negative value issues, but validate range above
             data = (data.astype(np.uint64) * (24 * 60 * 60)).astype("datetime64[s]")
         elif unit == "m":
             data = data.astype("datetime64[ms]")
```