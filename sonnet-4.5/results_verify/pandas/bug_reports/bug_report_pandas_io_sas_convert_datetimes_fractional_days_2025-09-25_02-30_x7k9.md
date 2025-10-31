# Bug Report: pandas.io.sas._convert_datetimes Fractional Days Truncation

**Target**: `pandas.io.sas.sas7bdat._convert_datetimes`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_convert_datetimes` function in `pandas.io.sas.sas7bdat` silently truncates fractional day values when converting SAS dates (unit="d"), causing data loss. SAS stores dates as float64 values representing days since 1960-01-01, which can include fractional days (e.g., 0.5 = noon), but the current implementation truncates these to whole days.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd
from pandas.io.sas.sas7bdat import _convert_datetimes

@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10))
def test_convert_datetimes_days_preserves_fractional_parts(x):
    series = pd.Series([x])
    result = _convert_datetimes(series, "d")

    if not pd.isna(result.iloc[0]):
        sas_origin = pd.Timestamp("1960-01-01")
        expected_timestamp = sas_origin + pd.Timedelta(days=x)
        actual_timestamp = result.iloc[0]

        time_diff_seconds = abs((expected_timestamp - actual_timestamp).total_seconds())
        assert time_diff_seconds < 1, f"Lost {time_diff_seconds} seconds of precision"
```

**Failing input**: `x=0.5` (and any fractional day value)

## Reproducing the Bug

```python
import numpy as np
import pandas as pd
from pandas import Timestamp

sas_origin = Timestamp("1960-01-01")
unix_origin = Timestamp("1970-01-01")
sas_datetimes = pd.Series([0.0, 0.5, 1.0, 1.5])
unit = "d"

td = (sas_origin - unix_origin).as_unit("s")
vals = np.array(sas_datetimes, dtype="M8[D]") + td
result = pd.Series(vals, dtype="M8[s]", index=sas_datetimes.index, copy=False)

print("Input (days):", sas_datetimes.tolist())
print("Output:", result.tolist())
print()
print("Expected for 0.5 days: 1960-01-01 12:00:00")
print("Actual for 0.5 days:  ", result.iloc[1])
print()
print("Expected for 1.5 days: 1960-01-02 12:00:00")
print("Actual for 1.5 days:  ", result.iloc[3])
```

## Why This Is A Bug

1. **Data Loss**: SAS stores dates as float64, allowing fractional day values. When reading SAS files with fractional dates, this function silently truncates them, losing precision.

2. **Silent Corruption**: The truncation happens without warning, so users have no indication that their data has been modified.

3. **Inconsistency**: The companion function `_parse_datetime` correctly handles fractional days using `timedelta(days=sas_datetime)`, which preserves fractional parts. The vectorized `_convert_datetimes` should behave consistently.

4. **Real-world Impact**: SAS DATE formats can represent dates with time-of-day precision when needed. For example, a SAS date value of 18262.5 represents 2010-01-01 12:00:00 (noon), but pandas would read it as 2010-01-01 00:00:00 (midnight).

## Fix

The bug is on line 107 of `sas7bdat.py`. The issue is using `dtype="M8[D]"` which truncates to whole days. The fix is to convert days to a finer unit (seconds or milliseconds) before creating the datetime64 array:

```diff
diff --git a/pandas/io/sas/sas7bdat.py b/pandas/io/sas/sas7bdat.py
index 1234567..abcdefg 100644
--- a/pandas/io/sas/sas7bdat.py
+++ b/pandas/io/sas/sas7bdat.py
@@ -104,8 +104,9 @@ def _convert_datetimes(sas_datetimes: pd.Series, unit: str) -> pd.Series:
         dt64ms = millis.view("M8[ms]") + td
         return pd.Series(dt64ms, index=sas_datetimes.index, copy=False)
     else:
-        vals = np.array(sas_datetimes, dtype="M8[D]") + td
-        return pd.Series(vals, dtype="M8[s]", index=sas_datetimes.index, copy=False)
+        # Convert days to seconds to preserve fractional days
+        vals = sas_datetimes._values * 86400  # days to seconds
+        dt64s = vals.astype('int64').view("M8[s]") + td
+        return pd.Series(dt64s, index=sas_datetimes.index, copy=False)
```