# Bug Report: pandas.io.sas._parse_datetime OverflowError on Large Date Values

**Target**: `pandas.io.sas.sas7bdat._parse_datetime`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `_parse_datetime` function crashes with an OverflowError when given large but otherwise valid SAS datetime values. This occurs because Python's datetime type cannot represent years beyond 9999, but the function doesn't validate input ranges or handle this limitation.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.io.sas.sas7bdat import _parse_datetime


@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10))
@settings(max_examples=1000)
def test_parse_datetime_days_unit(sas_datetime):
    result = _parse_datetime(sas_datetime, unit='d')

    if not pd.isna(result):
        assert isinstance(result, datetime)
```

**Failing input**: `sas_datetime=2936550.0, unit='d'`

## Reproducing the Bug

```python
from pandas.io.sas.sas7bdat import _parse_datetime
from datetime import datetime

sas_datetime = 2936550.0

result = _parse_datetime(sas_datetime, unit='d')
```

Output:
```
OverflowError: date value out of range
```

## Why This Is A Bug

The function attempts to create a datetime by adding `sas_datetime` days to January 1, 1960. However:

1. Python's datetime.max year is 9999 (December 31, 9999)
2. The maximum safe days from 1960-01-01 is approximately 2,936,549
3. Input values >= 2,936,550 cause an OverflowError
4. SAS can store dates beyond year 9999, making these values legitimate SAS data

The function handles NaN by returning `pd.NaT` (lines 68-69), but it doesn't validate numeric ranges. This means reading SAS files with far-future dates will crash rather than gracefully handling the limitation.

## Fix

```diff
--- a/pandas/io/sas/sas7bdat.py
+++ b/pandas/io/sas/sas7bdat.py
@@ -67,13 +67,21 @@ _sas_origin = Timestamp("1960-01-01")
 def _parse_datetime(sas_datetime: float, unit: str):
     if isna(sas_datetime):
         return pd.NaT

     if unit == "s":
-        return datetime(1960, 1, 1) + timedelta(seconds=sas_datetime)
+        try:
+            return datetime(1960, 1, 1) + timedelta(seconds=sas_datetime)
+        except (OverflowError, OSError):
+            return pd.NaT

     elif unit == "d":
-        return datetime(1960, 1, 1) + timedelta(days=sas_datetime)
+        try:
+            return datetime(1960, 1, 1) + timedelta(days=sas_datetime)
+        except (OverflowError, OSError):
+            return pd.NaT

     else:
         raise ValueError("unit must be 'd' or 's'")
```
