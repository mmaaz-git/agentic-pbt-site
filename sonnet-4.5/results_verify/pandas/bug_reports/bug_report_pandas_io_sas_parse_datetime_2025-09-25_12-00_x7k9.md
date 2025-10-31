# Bug Report: pandas.io.sas._parse_datetime OverflowError on Large Values

**Target**: `pandas.io.sas.sas7bdat._parse_datetime`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `_parse_datetime` function in `pandas.io.sas.sas7bdat` crashes with an `OverflowError` when given large but valid floating-point values that would result in dates outside Python's `datetime` range. The function accepts `float` values from SAS files but doesn't handle the case where these values exceed the representable date range.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.io.sas.sas7bdat import _parse_datetime

@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10))
def test_parse_datetime_handles_overflow(sas_days):
    result = _parse_datetime(sas_days, unit="d")
```

**Failing input**: `sas_days=2936550.0` (days), or `sas_seconds=1e15` (seconds)

## Reproducing the Bug

```python
from pandas.io.sas.sas7bdat import _parse_datetime

result = _parse_datetime(2936550.0, unit="d")
```

**Output**:
```
OverflowError: date value out of range
```

## Why This Is A Bug

The `_parse_datetime` function is designed to convert SAS date/datetime values (stored as floats representing days or seconds since 1960-01-01) to Python `datetime` objects. The function includes a check for `NaN` values (line 68) and returns `pd.NaT` in that case. However, it doesn't handle values that are numerically valid but exceed Python's datetime range.

Python's `datetime` type can represent dates from `datetime.min` (year 1) to `datetime.max` (year 9999). When adding a large timedelta to the SAS origin date (1960-01-01), values that would result in dates outside this range cause an `OverflowError`.

For example:
- `datetime(1960, 1, 1) + timedelta(days=2936550)` exceeds `datetime.max`
- This causes an unhandled crash instead of gracefully returning `pd.NaT`

SAS files can legitimately contain corrupt or extreme values, and the function should handle these gracefully rather than crashing. The similar `_convert_datetimes` function (lines 81-108) uses pandas operations that may handle this better, but `_parse_datetime` is a simpler utility that's more fragile.

## Fix

```diff
--- a/pandas/io/sas/sas7bdat.py
+++ b/pandas/io/sas/sas7bdat.py
@@ -68,12 +68,16 @@ def _parse_datetime(sas_datetime: float, unit: str):
     if isna(sas_datetime):
         return pd.NaT

-    if unit == "s":
-        return datetime(1960, 1, 1) + timedelta(seconds=sas_datetime)
-
-    elif unit == "d":
-        return datetime(1960, 1, 1) + timedelta(days=sas_datetime)
-
-    else:
+    try:
+        if unit == "s":
+            return datetime(1960, 1, 1) + timedelta(seconds=sas_datetime)
+        elif unit == "d":
+            return datetime(1960, 1, 1) + timedelta(days=sas_datetime)
+        else:
+            raise ValueError("unit must be 'd' or 's'")
+    except (OverflowError, ValueError):
+        return pd.NaT
+
+    if unit not in ("s", "d"):
         raise ValueError("unit must be 'd' or 's'")
```

Alternatively, a simpler fix that matches the existing code structure:

```diff
--- a/pandas/io/sas/sas7bdat.py
+++ b/pandas/io/sas/sas7bdat.py
@@ -68,10 +68,14 @@ def _parse_datetime(sas_datetime: float, unit: str):
     if isna(sas_datetime):
         return pd.NaT

-    if unit == "s":
-        return datetime(1960, 1, 1) + timedelta(seconds=sas_datetime)
-
-    elif unit == "d":
-        return datetime(1960, 1, 1) + timedelta(days=sas_datetime)
-
-    else:
+    try:
+        if unit == "s":
+            return datetime(1960, 1, 1) + timedelta(seconds=sas_datetime)
+        elif unit == "d":
+            return datetime(1960, 1, 1) + timedelta(days=sas_datetime)
+        else:
+            raise ValueError("unit must be 'd' or 's'")
+    except OverflowError:
+        return pd.NaT
+
+    raise ValueError("unit must be 'd' or 's'")
```