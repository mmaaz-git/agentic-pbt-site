# Bug Report: pandas.io.sas._convert_datetimes Fractional Days Truncation

**Target**: `pandas.io.sas.sas7bdat._convert_datetimes`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_convert_datetimes` function silently truncates fractional day components when unit='d', causing data loss and inconsistency with the `_parse_datetime` function.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd
from pandas.io.sas.sas7bdat import _parse_datetime, _convert_datetimes


@given(
    values=st.lists(
        st.floats(min_value=-1e8, max_value=1e8, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=20
    )
)
def test_convert_datetimes_consistency_with_parse_datetime(values):
    series = pd.Series(values)
    vectorized_result = _convert_datetimes(series, "d")

    for i, value in enumerate(values):
        scalar_result = _parse_datetime(value, "d")
        vectorized_value = vectorized_result.iloc[i]

        scalar_ts = pd.Timestamp(scalar_result)
        vectorized_ts = pd.Timestamp(vectorized_value)

        time_diff_ms = abs((scalar_ts - vectorized_ts).total_seconds() * 1000)

        assert time_diff_ms < 1, (
            f"Inconsistency at index {i} for value {value} with unit d: "
            f"_parse_datetime returned {scalar_result}, "
            f"_convert_datetimes returned {vectorized_value}, "
            f"difference: {time_diff_ms}ms"
        )
```

**Failing input**: `values=[0.5]`

## Reproducing the Bug

```python
import pandas as pd
from pandas.io.sas.sas7bdat import _parse_datetime, _convert_datetimes

value = 0.5

parse_result = _parse_datetime(value, "d")
print(f"_parse_datetime(0.5, 'd') = {parse_result}")

convert_result = _convert_datetimes(pd.Series([value]), "d").iloc[0]
print(f"_convert_datetimes([0.5], 'd') = {convert_result}")

diff = pd.Timestamp(parse_result) - pd.Timestamp(convert_result)
print(f"Difference: {diff}")
```

Output:
```
_parse_datetime(0.5, 'd') = 1960-01-01 12:00:00
_convert_datetimes([0.5], 'd') = 1960-01-01 00:00:00
Difference: 0 days 12:00:00
```

## Why This Is A Bug

1. **Data Loss**: The fractional day component (0.5 days = 12 hours) is silently truncated to 0, losing precision
2. **Inconsistency**: Two functions in the same module handling the same data type behave differently for identical inputs
3. **Silent Failure**: No warning or error is raised when data is truncated
4. **Violates Property**: The functions should be consistent - vectorized operations should match scalar operations

The root cause is in line 107 of `sas7bdat.py`:
```python
vals = np.array(sas_datetimes, dtype="M8[D]") + td
```

The `dtype="M8[D]"` (datetime64 with day precision) truncates any fractional day component, while `_parse_datetime` uses `timedelta(days=sas_datetime)` which preserves fractional days.

## Fix

```diff
--- a/pandas/io/sas/sas7bdat.py
+++ b/pandas/io/sas/sas7bdat.py
@@ -104,8 +104,11 @@ def _convert_datetimes(sas_datetimes: pd.Series, unit: str) -> pd.Series:
         dt64ms = millis.view("M8[ms]") + td
         return pd.Series(dt64ms, index=sas_datetimes.index, copy=False)
     else:
-        vals = np.array(sas_datetimes, dtype="M8[D]") + td
-        return pd.Series(vals, dtype="M8[s]", index=sas_datetimes.index, copy=False)
+        millis = cast_from_unit_vectorized(
+            sas_datetimes._values, unit="D", out_unit="ms"
+        )
+        dt64ms = millis.view("M8[ms]") + td
+        return pd.Series(dt64ms, index=sas_datetimes.index, copy=False)


 class _Column:
```

This makes the "d" (days) path behave like the "s" (seconds) path, preserving fractional components and maintaining consistency with `_parse_datetime`.