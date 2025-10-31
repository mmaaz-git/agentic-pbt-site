# Bug Report: pandas.io.sas _parse_datetime OverflowError

**Target**: `pandas.io.sas.sas7bdat._parse_datetime`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `_parse_datetime` function crashes with an `OverflowError` when processing large float values that exceed Python's `timedelta` limits, instead of handling them gracefully as invalid dates.

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
```

**Failing input**: `1e15` (seconds) or `1e15` (days)

## Reproducing the Bug

```python
from pandas.io.sas.sas7bdat import _parse_datetime

large_value = 1e15

result = _parse_datetime(large_value, 's')
```

**Output:**
```
OverflowError: date value out of range
```

**Same issue with days unit:**
```python
result = _parse_datetime(large_value, 'd')
```

**Output:**
```
OverflowError: date value out of range
```

## Why This Is A Bug

1. **Malformed SAS files**: Corrupt or malformed SAS files may contain arbitrary float values in datetime columns
2. **No documented limits**: The function does not document any input range constraints
3. **Inconsistent error handling**: The function handles NaN gracefully (returns `pd.NaT`) but crashes on large values
4. **Ungraceful failure**: An `OverflowError` is not a clear, user-friendly error for invalid date data

The function should either:
- Convert out-of-range values to `pd.NaT` (consistent with NaN handling)
- Raise a more descriptive `ValueError` with a helpful message
- Document the valid input range

## Fix

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
-
+    try:
+        if unit == "s":
+            return datetime(1960, 1, 1) + timedelta(seconds=sas_datetime)
+        elif unit == "d":
+            return datetime(1960, 1, 1) + timedelta(days=sas_datetime)
+    except (OverflowError, ValueError):
+        return pd.NaT
+
     else:
         raise ValueError("unit must be 'd' or 's'")
```

This fix makes the function handle out-of-range values consistently with how it handles NaN values, converting them to `pd.NaT` instead of crashing.