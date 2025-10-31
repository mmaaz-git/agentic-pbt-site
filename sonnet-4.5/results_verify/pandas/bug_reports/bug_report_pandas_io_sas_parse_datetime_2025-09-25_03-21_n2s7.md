# Bug Report: pandas.io.sas._parse_datetime Overflow Error

**Target**: `pandas.io.sas.sas7bdat._parse_datetime`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `_parse_datetime` function crashes with `OverflowError` when given large but valid float values, instead of returning `pd.NaT` like it does for `NaN` values. This is inconsistent error handling.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.io.sas.sas7bdat import _parse_datetime
import pandas as pd

@given(st.floats(allow_nan=False, allow_infinity=False))
def test_parse_datetime_days_no_crash(x):
    result = _parse_datetime(x, "d")
    assert result is not None or pd.isna(result)

@given(st.floats(allow_nan=False, allow_infinity=False))
def test_parse_datetime_seconds_no_crash(x):
    result = _parse_datetime(x, "s")
    assert result is not None or pd.isna(result)
```

**Failing input (days)**: `x=2936550.0`
**Failing input (seconds)**: `x=253717920000.0`

## Reproducing the Bug

```python
from pandas.io.sas.sas7bdat import _parse_datetime
import pandas as pd

result_nan = _parse_datetime(float('nan'), 'd')
print(f"_parse_datetime(nan, 'd') = {result_nan}")

try:
    result = _parse_datetime(2936550.0, 'd')
except OverflowError as e:
    print(f"OverflowError for days: {e}")

try:
    result = _parse_datetime(253717920000.0, 's')
except OverflowError as e:
    print(f"OverflowError for seconds: {e}")
```

## Why This Is A Bug

1. The function already handles invalid values (NaN) by returning `pd.NaT` (line 68-69 in sas7bdat.py)
2. Out-of-range values that cause `OverflowError` should be treated the same as NaN
3. This could occur with corrupted or malformed SAS files, and should be handled gracefully
4. Inconsistent error handling: NaN → NaT, but overflow → crash

## Fix

```diff
--- a/pandas/io/sas/sas7bdat.py
+++ b/pandas/io/sas/sas7bdat.py
@@ -67,11 +67,19 @@ def _parse_datetime(sas_datetime: float, unit: str):
 def _parse_datetime(sas_datetime: float, unit: str):
     if isna(sas_datetime):
         return pd.NaT

     if unit == "s":
-        return datetime(1960, 1, 1) + timedelta(seconds=sas_datetime)
-
+        try:
+            return datetime(1960, 1, 1) + timedelta(seconds=sas_datetime)
+        except (OverflowError, ValueError):
+            return pd.NaT
     elif unit == "d":
-        return datetime(1960, 1, 1) + timedelta(days=sas_datetime)
-
+        try:
+            return datetime(1960, 1, 1) + timedelta(days=sas_datetime)
+        except (OverflowError, ValueError):
+            return pd.NaT
     else:
         raise ValueError("unit must be 'd' or 's'")
```