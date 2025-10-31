# Bug Report: pandas.io.sas._convert_datetimes Missing Input Validation

**Target**: `pandas.io.sas.sas7bdat._convert_datetimes`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `_convert_datetimes` function accepts a `unit` parameter but does not validate it, silently treating any invalid value as "d" (days). This is inconsistent with the related `_parse_datetime` function which properly validates the unit parameter, and could lead to silent data corruption.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd
from pandas.io.sas.sas7bdat import _convert_datetimes

@given(
    st.lists(st.floats(min_value=0, max_value=1e6, allow_nan=False), min_size=1, max_size=100),
    st.text(min_size=1, max_size=10).filter(lambda x: x not in ["s", "d"])
)
def test_convert_datetimes_invalid_unit(values, invalid_unit):
    sas_dates = pd.Series(values, dtype=float)

    result = _convert_datetimes(sas_dates, invalid_unit)

    assert False, f"Should have raised ValueError for invalid unit '{invalid_unit}', but got result"
```

**Failing input**: Any invalid unit value, e.g., `unit="ms"`, `unit="invalid"`, `unit="h"`

## Reproducing the Bug

```python
import pandas as pd
from pandas.io.sas.sas7bdat import _convert_datetimes

sas_dates = pd.Series([0, 1, 100], dtype=float)

result = _convert_datetimes(sas_dates, "invalid_unit")
print(result)
```

**Output**:
```
0   1960-01-01
1   1960-01-02
2   1960-04-10
dtype: datetime64[s]
```

The function silently treats "invalid_unit" as "d" (days) instead of raising an error.

## Why This Is A Bug

The function has an if-else structure that treats any non-"s" value as "d":

```python
def _convert_datetimes(sas_datetimes: pd.Series, unit: str) -> pd.Series:
    td = (_sas_origin - _unix_origin).as_unit("s")
    if unit == "s":
        # seconds conversion
        ...
    else:
        # days conversion - executed for ANY non-"s" value!
        ...
```

This creates several problems:

1. **Inconsistency**: The related function `_parse_datetime` validates the unit parameter:
   ```python
   def _parse_datetime(sas_datetime: float, unit: str):
       if unit == "s":
           ...
       elif unit == "d":
           ...
       else:
           raise ValueError("unit must be 'd' or 's'")
   ```

2. **Silent failure**: Invalid units don't raise errors, potentially corrupting data
3. **API contract violation**: The docstring states the parameter should be "{'d', 's'}" but doesn't enforce it
4. **Future-proofing**: If someone tries to add support for other units (e.g., "ms"), they could accidentally trigger this bug

While current internal usage only passes "s" or "d", the lack of validation makes the code fragile and violates defensive programming principles.

## Fix

```diff
--- a/pandas/io/sas/sas7bdat.py
+++ b/pandas/io/sas/sas7bdat.py
@@ -81,11 +81,13 @@ def _convert_datetimes(sas_datetimes: pd.Series, unit: str) -> pd.Series:
     Series
        Series of datetime64 dtype or datetime.datetime.
     """
+    if unit not in ("s", "d"):
+        raise ValueError("unit must be 'd' or 's'")
+
     td = (_sas_origin - _unix_origin).as_unit("s")
     if unit == "s":
         millis = cast_from_unit_vectorized(
             sas_datetimes._values, unit="s", out_unit="ms"
         )
         dt64ms = millis.view("M8[ms]") + td
         return pd.Series(dt64ms, index=sas_datetimes.index, copy=False)
```

This makes the function consistent with `_parse_datetime` and follows the principle of failing fast with clear error messages.