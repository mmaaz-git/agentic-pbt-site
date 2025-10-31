# Bug Report: pandas.io.sas._parse_datetime OverflowError

**Target**: `pandas.io.sas.sas7bdat._parse_datetime`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`_parse_datetime` crashes with `OverflowError` when given large day values that result in dates beyond Python's `datetime.max` (year 9999).

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.io.sas.sas7bdat import _parse_datetime

@given(
    st.floats(
        min_value=-1e8,
        max_value=1e8,
        allow_nan=False,
        allow_infinity=False
    )
)
@settings(max_examples=200)
def test_parse_datetime_days_monotonicity(days):
    smaller = days
    larger = days + 1.0

    result_smaller = _parse_datetime(smaller, 'd')
    result_larger = _parse_datetime(larger, 'd')

    assert result_larger > result_smaller
```

**Failing input**: `days=2936549.0`

## Reproducing the Bug

```python
from pandas.io.sas.sas7bdat import _parse_datetime

days = 2936549.0
result = _parse_datetime(days, 'd')
```

**Output:**
```
OverflowError: date value out of range
```

**Explanation:**
- Input: 2936549 days from 1960-01-01
- That's approximately 8041 years from 1960
- Final year would be: 10001
- Python's `datetime.max.year` is 9999
- The calculation `datetime(1960, 1, 1) + timedelta(days=2936549)` overflows

## Why This Is A Bug

1. **Crashes on valid input**: SAS date values can theoretically be any float, and the function should handle out-of-range values gracefully instead of crashing
2. **Inconsistent behavior**: Similar pandas datetime functions typically return `pd.NaT` for out-of-range values rather than raising
3. **Poor error handling**: The function checks for NaT input but doesn't validate the range of the date value

The function is documented to handle SAS datetime values but crashes when given values that are technically valid SAS dates but outside Python's datetime range.

## Fix

```diff
--- a/pandas/io/sas/sas7bdat.py
+++ b/pandas/io/sas/sas7bdat.py
@@ -67,13 +67,19 @@ def _parse_datetime(sas_datetime: float, unit: str):
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
+    except (OverflowError, ValueError) as e:
+        if isinstance(e, ValueError) and "unit must be" in str(e):
+            raise
+        # Return NaT for out-of-range datetime values
+        return pd.NaT
+    except Exception:
         raise ValueError("unit must be 'd' or 's'")
```