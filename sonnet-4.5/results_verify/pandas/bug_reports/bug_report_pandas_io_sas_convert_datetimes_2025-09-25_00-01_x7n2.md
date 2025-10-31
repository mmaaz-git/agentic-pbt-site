# Bug Report: pandas.io.sas._convert_datetimes Overflow on Very Large Values

**Target**: `pandas.io.sas.sas7bdat._convert_datetimes`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `_convert_datetimes` function in `pandas.io.sas.sas7bdat` crashes with an `OverflowError` or `OutOfBoundsDatetime` exception when given very large float values that cannot be represented in pandas datetime64 format. This can occur when reading SAS files with invalid or corrupted datetime columns.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st, settings
from pandas.io.sas.sas7bdat import _convert_datetimes


@given(st.lists(st.floats(allow_nan=False, allow_infinity=False,
                          min_value=1e16, max_value=1e18), min_size=1, max_size=10))
@settings(max_examples=500)
def test_convert_datetimes_very_large_should_not_crash(values):
    series = pd.Series(values)
    result = _convert_datetimes(series, "s")
    assert isinstance(result, pd.Series)
```

**Failing input**: `[1e+16]`

## Reproducing the Bug

```python
import pandas as pd
from pandas.io.sas.sas7bdat import _convert_datetimes

series = pd.Series([1e16])
result = _convert_datetimes(series, "s")
```

Output:
```
pandas._libs.tslibs.np_datetime.OutOfBoundsDatetime: cannot convert input 1e+16 with the unit 's'
```

For even larger values:
```python
series = pd.Series([9.223372036854776e+18])
result = _convert_datetimes(series, "s")
```

Output:
```
OverflowError: Python int too large to convert to C long
```

## Why This Is A Bug

The function crashes instead of gracefully handling out-of-range values. While SAS files typically contain reasonable datetime values, the function could encounter:
1. Corrupted data in damaged SAS files
2. Special marker values that represent missing or invalid data
3. Edge cases in legacy SAS formats

The function already handles NaN/NaT values correctly (preserving them), but fails to handle values that are numerically valid floats but outside the representable datetime range. This inconsistency makes error handling difficult for users.

## Fix

Add bounds checking before converting values. Since pandas datetime64 has a limited range (approximately 1677-2262 for nanosecond resolution, but wider for other resolutions), we should catch overflow errors and convert out-of-range values to NaT:

```diff
diff --git a/pandas/io/sas/sas7bdat.py b/pandas/io/sas/sas7bdat.py
index 1234567..abcdefg 100644
--- a/pandas/io/sas/sas7bdat.py
+++ b/pandas/io/sas/sas7bdat.py
@@ -98,13 +98,25 @@ def _convert_datetimes(sas_datetimes: pd.Series, unit: str) -> pd.Series:
     """
     td = (_sas_origin - _unix_origin).as_unit("s")
     if unit == "s":
-        millis = cast_from_unit_vectorized(
-            sas_datetimes._values, unit="s", out_unit="ms"
-        )
-        dt64ms = millis.view("M8[ms]") + td
-        return pd.Series(dt64ms, index=sas_datetimes.index, copy=False)
+        try:
+            millis = cast_from_unit_vectorized(
+                sas_datetimes._values, unit="s", out_unit="ms"
+            )
+            dt64ms = millis.view("M8[ms]") + td
+            return pd.Series(dt64ms, index=sas_datetimes.index, copy=False)
+        except (OverflowError, OutOfBoundsDatetime):
+            # Handle out-of-range values by converting them to NaT
+            result = sas_datetimes.copy()
+            result = result.apply(lambda x: _parse_datetime(x, "s") if not isna(x) else pd.NaT)
+            return pd.Series(result, index=sas_datetimes.index)
     else:
-        vals = np.array(sas_datetimes, dtype="M8[D]") + td
-        return pd.Series(vals, dtype="M8[s]", index=sas_datetimes.index, copy=False)
+        try:
+            vals = np.array(sas_datetimes, dtype="M8[D]") + td
+            return pd.Series(vals, dtype="M8[s]", index=sas_datetimes.index, copy=False)
+        except (OverflowError, OutOfBoundsDatetime):
+            # Handle out-of-range values by converting them to NaT
+            result = sas_datetimes.copy()
+            result = result.apply(lambda x: _parse_datetime(x, "d") if not isna(x) else pd.NaT)
+            return pd.Series(result, index=sas_datetimes.index)
```

Note: The fix above would also require updating `_parse_datetime` to handle overflow (as described in the companion bug report), or alternatively, implementing direct bounds checking in `_convert_datetimes`.