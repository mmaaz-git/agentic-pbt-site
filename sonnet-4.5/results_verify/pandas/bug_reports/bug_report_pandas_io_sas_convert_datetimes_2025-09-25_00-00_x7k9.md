# Bug Report: pandas.io.sas._convert_datetimes OverflowError on Large Values

**Target**: `pandas.io.sas.sas7bdat._convert_datetimes`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `_convert_datetimes` function crashes with an OverflowError when processing extremely large datetime values instead of handling them gracefully (e.g., returning NaT or raising a descriptive ValueError).

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st, settings
from pandas.io.sas.sas7bdat import _convert_datetimes


@given(value=st.floats(min_value=1e15, max_value=1e20, allow_nan=False, allow_infinity=False))
@settings(max_examples=50)
def test_convert_datetimes_extreme_values_seconds(value):
    series = pd.Series([value])
    result = _convert_datetimes(series, 's')
    assert len(result) == 1
```

**Failing input**: `1e20` (or any sufficiently large float)

## Reproducing the Bug

```python
import pandas as pd
from pandas.io.sas.sas7bdat import _convert_datetimes

series = pd.Series([1e20])
result = _convert_datetimes(series, 's')
```

Output:
```
OverflowError: Python int too large to convert to C long
```

## Why This Is A Bug

The function accepts a float parameter without documented range restrictions, but crashes on large values instead of:
1. Validating input and raising a descriptive error, or
2. Returning NaT for out-of-range values, or
3. Clamping to valid datetime ranges

This function is called in production code at `sas7bdat.py:736-738` when reading SAS files. If a corrupted or malformed SAS file contains extreme datetime values, the entire read operation will crash with an unhelpful OverflowError instead of handling the invalid data gracefully.

## Fix

Add input validation to check for overflow before conversion:

```diff
diff --git a/pandas/io/sas/sas7bdat.py b/pandas/io/sas/sas7bdat.py
index 1234567..abcdefg 100644
--- a/pandas/io/sas/sas7bdat.py
+++ b/pandas/io/sas/sas7bdat.py
@@ -98,6 +98,13 @@ def _convert_datetimes(sas_datetimes: pd.Series, unit: str) -> pd.Series:
     """
     td = (_sas_origin - _unix_origin).as_unit("s")
     if unit == "s":
+        # Check for overflow: max timedelta is ~2.9e11 seconds
+        max_seconds = 9223372036
+        overflow_mask = (sas_datetimes.abs() > max_seconds)
+        if overflow_mask.any():
+            sas_datetimes = sas_datetimes.copy()
+            sas_datetimes[overflow_mask] = pd.NA
+
         millis = cast_from_unit_vectorized(
             sas_datetimes._values, unit="s", out_unit="ms"
         )
```

Alternatively, wrap the conversion in a try-except block and handle OverflowError by returning NaT for problematic values.