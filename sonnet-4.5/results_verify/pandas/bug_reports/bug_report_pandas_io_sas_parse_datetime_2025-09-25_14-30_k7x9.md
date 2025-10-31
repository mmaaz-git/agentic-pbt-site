# Bug Report: pandas.io.sas._parse_datetime OverflowError

**Target**: `pandas.io.sas.sas7bdat._parse_datetime`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `_parse_datetime` function crashes with `OverflowError` when given SAS date values that exceed Python's datetime range (approximately 8040 years from 1960-01-01), even though such values may be valid SAS dates.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from hypothesis import given, settings, strategies as st
from pandas.io.sas.sas7bdat import _parse_datetime

@given(
    sas_datetime=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10),
    unit=st.sampled_from(["s", "d"])
)
@settings(max_examples=1000)
def test_parse_datetime_determinism(sas_datetime, unit):
    result1 = _parse_datetime(sas_datetime, unit)
    result2 = _parse_datetime(sas_datetime, unit)
    assert result1 == result2
```

**Failing input**: `sas_datetime=2936550.0, unit='d'`

## Reproducing the Bug

```python
from datetime import datetime, timedelta
from pandas.io.sas.sas7bdat import _parse_datetime

sas_datetime = 2936550.0

result = _parse_datetime(sas_datetime, 'd')
```

Output:
```
OverflowError: date value out of range
```

## Why This Is A Bug

The function accepts a float as input without validating that it falls within Python's datetime range. The value 2936550.0 days from 1960-01-01 represents a date in year 10000, which exceeds datetime's maximum year of 9999.

While this may represent an unrealistic date, the function should either:
1. Validate inputs and raise an informative error message, or
2. Handle overflow gracefully by returning NaT or using a wider datetime type

The sister function `_convert_datetimes` handles the same input correctly by using numpy's datetime64 type which has a wider range.

## Fix

The function appears to be legacy code (unused in the codebase). The fix is to either:

**Option 1**: Remove the function entirely since `_convert_datetimes` is used instead:

```diff
- def _parse_datetime(sas_datetime: float, unit: str):
-     if isna(sas_datetime):
-         return pd.NaT
-
-     if unit == "s":
-         return datetime(1960, 1, 1) + timedelta(seconds=sas_datetime)
-
-     elif unit == "d":
-         return datetime(1960, 1, 1) + timedelta(days=sas_datetime)
-
-     else:
-         raise ValueError("unit must be 'd' or 's'")
```

**Option 2**: Add validation with an informative error message:

```diff
 def _parse_datetime(sas_datetime: float, unit: str):
     if isna(sas_datetime):
         return pd.NaT

+    # Python datetime range: 0001-01-01 to 9999-12-31
+    # From 1960-01-01: approximately -2,934,235 to 2,936,549 days
+    MAX_DAYS = 2_936_549
+    MIN_DAYS = -2_934_235
+    MAX_SECONDS = MAX_DAYS * 86400
+    MIN_SECONDS = MIN_DAYS * 86400
+
     if unit == "s":
+        if not (MIN_SECONDS <= sas_datetime <= MAX_SECONDS):
+            raise ValueError(f"SAS datetime {sas_datetime} seconds from 1960-01-01 "
+                           f"is outside Python datetime range")
         return datetime(1960, 1, 1) + timedelta(seconds=sas_datetime)

     elif unit == "d":
+        if not (MIN_DAYS <= sas_datetime <= MAX_DAYS):
+            raise ValueError(f"SAS date {sas_datetime} days from 1960-01-01 "
+                           f"is outside Python datetime range")
         return datetime(1960, 1, 1) + timedelta(days=sas_datetime)

     else:
         raise ValueError("unit must be 'd' or 's'")
```