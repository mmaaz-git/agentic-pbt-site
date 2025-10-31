# Bug Report: pandas.io.sas._convert_datetimes Overflow Error

**Target**: `pandas.io.sas.sas7bdat._convert_datetimes`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `_convert_datetimes` function (vectorized datetime converter) crashes with `OverflowError` or `OutOfBoundsDatetime` when given large float values, instead of handling them gracefully like `_parse_datetime` should do for NaN values.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.io.sas.sas7bdat import _convert_datetimes
import pandas as pd

@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=100)
)
def test_convert_datetimes_seconds_no_crash(values):
    series = pd.Series(values)
    result = _convert_datetimes(series, "s")
    assert len(result) == len(values)
```

**Failing input**: `values=[9.223372036854776e+18]`

## Reproducing the Bug

```python
import pandas as pd
from pandas.io.sas.sas7bdat import _convert_datetimes

series_small = pd.Series([0.0, 100.0, 1000.0])
result = _convert_datetimes(series_small, "s")
print(f"Small values work: {result.tolist()}")

series_large = pd.Series([9.223372036854776e+18])
try:
    result = _convert_datetimes(series_large, "s")
except OverflowError as e:
    print(f"OverflowError: {e}")
```

## Why This Is A Bug

1. This is the vectorized version of `_parse_datetime`, which has the same overflow issue
2. Like `_parse_datetime`, it should handle out-of-range values gracefully by returning NaT
3. This could occur with corrupted SAS files containing invalid datetime values
4. Users expect consistent behavior - NaN inputs should produce NaT, overflow values should too

## Fix

```diff
--- a/pandas/io/sas/sas7bdat.py
+++ b/pandas/io/sas/sas7bdat.py
@@ -98,12 +98,17 @@ def _convert_datetimes(sas_datetimes: pd.Series, unit: str) -> pd.Series:
        Series of datetime64 dtype or datetime.datetime.
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
+        except (OverflowError, ValueError, pd._libs.tslibs.np_datetime.OutOfBoundsDatetime):
+            return pd.Series([pd.NaT] * len(sas_datetimes), index=sas_datetimes.index)
     else:
-        vals = np.array(sas_datetimes, dtype="M8[D]") + td
-        return pd.Series(vals, dtype="M8[s]", index=sas_datetimes.index, copy=False)
+        try:
+            vals = np.array(sas_datetimes, dtype="M8[D]") + td
+            return pd.Series(vals, dtype="M8[s]", index=sas_datetimes.index, copy=False)
+        except (OverflowError, ValueError):
+            return pd.Series([pd.NaT] * len(sas_datetimes), index=sas_datetimes.index)
```