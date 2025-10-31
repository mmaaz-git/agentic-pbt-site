# Bug Report: pandas.io.sas._convert_datetimes Silently Accepts Invalid Unit Values

**Target**: `pandas.io.sas.sas7bdat._convert_datetimes`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The internal function `_convert_datetimes` accepts any string value for the `unit` parameter and silently treats non-'s' values as 'd' (days), violating its documented contract that specifies only {'d', 's'} as valid values.

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

if __name__ == "__main__":
    test_convert_datetimes_invalid_unit_raises()
```

<details>

<summary>
**Failing input**: `invalid_unit=''` (empty string)
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 17, in <module>
    test_convert_datetimes_invalid_unit_raises()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 6, in test_convert_datetimes_invalid_unit_raises
    @settings(max_examples=20)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 12, in test_convert_datetimes_invalid_unit_raises
    assert False, f"Should have raised error for invalid unit '{invalid_unit}'"
           ^^^^^
AssertionError: Should have raised error for invalid unit ''
Falsifying example: test_convert_datetimes_invalid_unit_raises(
    invalid_unit='',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
from pandas.io.sas.sas7bdat import _convert_datetimes

# Create a simple test series with floating point values
series = pd.Series([1.0, 2.0, 3.0])

# Test with an empty string as the unit parameter (invalid)
result = _convert_datetimes(series, '')

print(f"Result dtype: {result.dtype}")
print(f"Result values: {result.values}")
print(f"Result: {result}")
```

<details>

<summary>
Function accepts empty string and treats it as 'd' without error
</summary>
```
Result dtype: datetime64[s]
Result values: ['1960-01-02T00:00:00' '1960-01-03T00:00:00' '1960-01-04T00:00:00']
Result: 0   1960-01-02
1   1960-01-03
2   1960-01-04
dtype: datetime64[s]
```
</details>

## Why This Is A Bug

The function's docstring explicitly documents `unit : {'d', 's'}` using Python's standard set notation, which conventionally means these are the only two acceptable values. However, the implementation uses an `if unit == "s": ... else:` pattern that treats ANY non-'s' value as 'd', including empty strings, typos like 'day' or 'D', and arbitrary invalid inputs.

This violates the principle of explicit parameter validation and can mask programming errors. A developer might accidentally pass 'D' instead of 'd' or use 'days' thinking it's more readable, and the function would silently produce results using the wrong unit interpretation rather than alerting them to the mistake. This is particularly problematic in data processing pipelines where silent failures can propagate incorrect datetime conversions throughout the analysis.

## Relevant Context

The function is located at `/pandas/io/sas/sas7bdat.py` lines 81-108. It's an internal function (underscore prefix) primarily used by pandas' SAS file reader for converting SAS datetime formats. The function converts SAS date/datetime floats to pandas datetime64 objects, with SAS using a different epoch (1960-01-01) than Unix timestamps.

The implementation shows:
- Line 100: Checks `if unit == "s"` to handle seconds
- Line 106: Uses `else:` which catches everything including invalid values
- No validation that the unit is actually 'd' when not 's'

This is inconsistent with Python's "errors should never pass silently" principle and pandas' general approach of validating parameters in other datetime conversion functions.

Documentation: https://pandas.pydata.org/docs/reference/api/pandas.read_sas.html (parent function that uses this internally)

## Proposed Fix

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
+        raise ValueError(f"unit must be 'd' or 's', got '{unit}'")
```