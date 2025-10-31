# Bug Report: pandas.io.sas._convert_datetimes Missing Unit Validation

**Target**: `pandas.io.sas.sas7bdat._convert_datetimes`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`_convert_datetimes` silently treats any invalid unit value as 'd' (days) instead of validating the unit parameter and raising an error. This is inconsistent with `_parse_datetime` which correctly validates the unit parameter.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd
from pandas.io.sas.sas7bdat import _convert_datetimes

@given(st.text().filter(lambda x: x not in ['d', 's']))
@settings(max_examples=20)
def test_convert_datetimes_invalid_unit_raises(invalid_unit):
    series = pd.Series([1.0, 2.0, 3.0])

    try:
        _convert_datetimes(series, invalid_unit)
        assert False, f"Should have raised error for invalid unit '{invalid_unit}'"
    except (ValueError, KeyError) as e:
        pass
```

**Failing input**: `invalid_unit=''` (empty string)

## Reproducing the Bug

```python
import pandas as pd
from pandas.io.sas.sas7bdat import _convert_datetimes

series = pd.Series([1.0, 2.0, 3.0])

result = _convert_datetimes(series, '')
print(f"Result dtype: {result.dtype}")
print(f"Result values: {result.values}")
```

**Output:**
```
Result dtype: datetime64[s]
Result values: ['1960-01-02T00:00:00' '1960-01-03T00:00:00' '1960-01-04T00:00:00']
```

The function accepts empty string (and any invalid unit) and treats it as 'd' (days) without raising an error.

## Why This Is A Bug

1. **API contract violation**: The docstring clearly states `unit : {'d', 's'}` indicating only these two values are valid
2. **Inconsistent behavior**: `_parse_datetime` correctly raises `ValueError` for invalid units, but `_convert_datetimes` silently accepts them
3. **Silent failure**: Invalid input is processed without error, making bugs harder to detect
4. **Misleading results**: Users might pass typos like ' d' or 'day' and get unexpected results

Looking at the source code sas7bdat.py:81-108:
```python
def _convert_datetimes(sas_datetimes: pd.Series, unit: str) -> pd.Series:
    """
    unit : {'d', 's'}
       "d" if the floats represent dates, "s" for datetimes
    """
    td = (_sas_origin - _unix_origin).as_unit("s")
    if unit == "s":
        # ... handle seconds
    else:
        # BUG: Treats EVERYTHING else as 'd', including invalid units
        vals = np.array(sas_datetimes, dtype="M8[D]") + td
        return pd.Series(vals, dtype="M8[s]", index=sas_datetimes.index, copy=False)
```

## Fix

```diff
--- a/pandas/io/sas/sas7bdat.py
+++ b/pandas/io/sas/sas7bdat.py
@@ -98,10 +98,13 @@ def _convert_datetimes(sas_datetimes: pd.Series, unit: str) -> pd.Series:
     """
     td = (_sas_origin - _unix_origin).as_unit("s")
     if unit == "s":
         millis = cast_from_unit_vectorized(
             sas_datetimes._values, unit="s", out_unit="ms"
         )
         dt64ms = millis.view("M8[ms]") + td
         return pd.Series(dt64ms, index=sas_datetimes.index, copy=False)
-    else:
+    elif unit == "d":
         vals = np.array(sas_datetimes, dtype="M8[D]") + td
         return pd.Series(vals, dtype="M8[s]", index=sas_datetimes.index, copy=False)
+    else:
+        raise ValueError("unit must be 'd' or 's'")
```