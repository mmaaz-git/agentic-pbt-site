# Bug Report: pandas.io.sas._parse_datetime Overflow on Large Values

**Target**: `pandas.io.sas.sas7bdat._parse_datetime`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `_parse_datetime` function in `pandas.io.sas.sas7bdat` crashes with an `OverflowError` when given moderately large datetime values that exceed Python's `datetime` and `timedelta` limits. This can occur when reading SAS files with invalid or corrupted datetime values.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.io.sas.sas7bdat import _parse_datetime


@given(st.floats(allow_nan=False, allow_infinity=False,
                 min_value=1e14, max_value=1e17))
@settings(max_examples=500)
def test_parse_datetime_large_seconds_should_not_crash(sas_datetime):
    result = _parse_datetime(sas_datetime, "s")
    assert result is not None


@given(st.floats(allow_nan=False, allow_infinity=False,
                 min_value=1e6, max_value=1e8))
@settings(max_examples=500)
def test_parse_datetime_large_days_should_not_crash(sas_datetime):
    result = _parse_datetime(sas_datetime, "d")
    assert result is not None
```

**Failing input (seconds)**: `100000000000000.0`
**Failing input (days)**: `2936550.0`

## Reproducing the Bug

```python
import pandas as pd
from pandas.io.sas.sas7bdat import _parse_datetime

result = _parse_datetime(100000000000000.0, "s")
```

Output:
```
OverflowError: days=1157407407; must have magnitude <= 999999999
```

```python
result = _parse_datetime(2936550.0, "d")
```

Output:
```
OverflowError: date value out of range
```

## Why This Is A Bug

The function crashes instead of gracefully handling out-of-range values. While SAS files typically contain reasonable datetime values, the function could encounter corrupted data or special marker values. The function should either:
1. Return `pd.NaT` for out-of-range values (similar to how it handles NaN input)
2. Raise a more specific, catchable exception with a helpful error message

The current behavior violates the principle of graceful degradation and makes it difficult for users to handle corrupted SAS files.

## Fix

Add bounds checking before attempting to create the datetime:

```diff
diff --git a/pandas/io/sas/sas7bdat.py b/pandas/io/sas/sas7bdat.py
index 1234567..abcdefg 100644
--- a/pandas/io/sas/sas7bdat.py
+++ b/pandas/io/sas/sas7bdat.py
@@ -67,11 +67,21 @@ _sas_origin = Timestamp("1960-01-01")
 def _parse_datetime(sas_datetime: float, unit: str):
     if isna(sas_datetime):
         return pd.NaT

+    # Check for values that would overflow datetime/timedelta
+    if unit == "s":
+        # timedelta max is 999999999 days
+        max_seconds = 999999999 * 86400
+        if abs(sas_datetime) > max_seconds:
+            return pd.NaT
+    elif unit == "d":
+        # datetime supports years 1-9999, approximately ±3,652,425 days from 1960
+        if abs(sas_datetime) > 3000000:
+            return pd.NaT
+
     if unit == "s":
         return datetime(1960, 1, 1) + timedelta(seconds=sas_datetime)

     elif unit == "d":
         return datetime(1960, 1, 1) + timedelta(days=sas_datetime)
```