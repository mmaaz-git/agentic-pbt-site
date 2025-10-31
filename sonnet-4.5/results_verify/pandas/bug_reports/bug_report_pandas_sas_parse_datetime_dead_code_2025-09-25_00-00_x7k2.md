# Bug Report: pandas.io.sas._parse_datetime Dead Code with OverflowError

**Target**: `pandas.io.sas.sas7bdat._parse_datetime`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_parse_datetime` function in `pandas/io/sas/sas7bdat.py` is dead code (defined but never called) and contains an unhandled OverflowError bug for large input values that would fail if anyone attempted to use it.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from pandas.io.sas.sas7bdat import _parse_datetime
from hypothesis import given, strategies as st, assume

@given(st.floats(allow_nan=False, allow_infinity=False))
def test_parse_datetime_seconds_increases_monotonically(sas_datetime):
    assume(sas_datetime >= 0)
    assume(sas_datetime < 1e15)

    dt1 = _parse_datetime(sas_datetime, "s")
    dt2 = _parse_datetime(sas_datetime + 1, "s")

    if not pd.isna(dt1) and not pd.isna(dt2):
        assert dt2 > dt1
```

**Failing input**: `sas_datetime=253717919999.0`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from pandas.io.sas.sas7bdat import _parse_datetime

try:
    result = _parse_datetime(253717920000.0, "s")
except OverflowError as e:
    print(f"OverflowError: {e}")

try:
    result = _parse_datetime(2936550.0, "d")
except OverflowError as e:
    print(f"OverflowError: {e}")
```

## Why This Is A Bug

The function has two issues:

1. **Dead Code**: `_parse_datetime` is defined in `sas7bdat.py:67` but never called anywhere in the pandas codebase. The actual datetime conversion is handled by `_convert_datetimes` which uses numpy vectorized operations.

2. **Latent OverflowError**: If anyone did try to use this function, it would crash with `OverflowError: date value out of range` for large but valid float inputs when computing `datetime(1960, 1, 1) + timedelta(seconds=sas_datetime)` or the days equivalent.

The function should either:
- Be removed entirely (recommended, as it's unused dead code)
- Be fixed to handle the full range of valid SAS datetime values

## Fix

Remove the dead code entirely, as `_convert_datetimes` already handles this functionality correctly using vectorized operations:

```diff
diff --git a/pandas/io/sas/sas7bdat.py b/pandas/io/sas/sas7bdat.py
index 1234567..abcdefg 100644
--- a/pandas/io/sas/sas7bdat.py
+++ b/pandas/io/sas/sas7bdat.py
@@ -64,17 +64,6 @@ _unix_origin = Timestamp("1970-01-01")
 _sas_origin = Timestamp("1960-01-01")


-def _parse_datetime(sas_datetime: float, unit: str):
-    if isna(sas_datetime):
-        return pd.NaT
-
-    if unit == "s":
-        return datetime(1960, 1, 1) + timedelta(seconds=sas_datetime)
-
-    elif unit == "d":
-        return datetime(1960, 1, 1) + timedelta(days=sas_datetime)
-
-    else:
-        raise ValueError("unit must be 'd' or 's'")
-
-
 def _convert_datetimes(sas_datetimes: pd.Series, unit: str) -> pd.Series:
```