# Bug Report: pandas.io.sas._convert_datetimes OverflowError on Large Float Values

**Target**: `pandas.io.sas.sas7bdat._convert_datetimes`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `_convert_datetimes` function crashes with `OverflowError` or `OutOfBoundsDatetime` when processing large float values that exceed the valid datetime range, instead of gracefully handling them by returning NaT (Not a Time) values.

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

# Run the test
if __name__ == "__main__":
    test_convert_datetimes_seconds_no_crash()
```

<details>

<summary>
**Failing input**: `values=[9.223372036854776e+18]`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 15, in <module>
  |     test_convert_datetimes_seconds_no_crash()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 6, in test_convert_datetimes_seconds_no_crash
  |     st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=100)
  |                ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 10, in test_convert_datetimes_seconds_no_crash
    |     result = _convert_datetimes(series, "s")
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/sas/sas7bdat.py", line 101, in _convert_datetimes
    |     millis = cast_from_unit_vectorized(
    |         sas_datetimes._values, unit="s", out_unit="ms"
    |     )
    |   File "pandas/_libs/tslibs/conversion.pyx", line 145, in pandas._libs.tslibs.conversion.cast_from_unit_vectorized
    | OverflowError: Python int too large to convert to C long
    | Falsifying example: test_convert_datetimes_seconds_no_crash(
    |     values=[9.223372036854776e+18],
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "pandas/_libs/tslibs/conversion.pyx", line 156, in pandas._libs.tslibs.conversion.cast_from_unit_vectorized
    | OverflowError: value too large
    |
    | The above exception was the direct cause of the following exception:
    |
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 10, in test_convert_datetimes_seconds_no_crash
    |     result = _convert_datetimes(series, "s")
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/sas/sas7bdat.py", line 101, in _convert_datetimes
    |     millis = cast_from_unit_vectorized(
    |         sas_datetimes._values, unit="s", out_unit="ms"
    |     )
    |   File "pandas/_libs/tslibs/conversion.pyx", line 160, in pandas._libs.tslibs.conversion.cast_from_unit_vectorized
    | pandas._libs.tslibs.np_datetime.OutOfBoundsDatetime: cannot convert input 9223372036854776.0 with the unit 's'
    | Falsifying example: test_convert_datetimes_seconds_no_crash(
    |     values=[9223372036854776.0],
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
import pandas as pd
from pandas.io.sas.sas7bdat import _convert_datetimes

# Test with small values that should work
series_small = pd.Series([0.0, 100.0, 1000.0])
result = _convert_datetimes(series_small, "s")
print(f"Small values work: {result.tolist()}")

# Test with large value that causes overflow
series_large = pd.Series([9.223372036854776e+18])
try:
    result = _convert_datetimes(series_large, "s")
    print(f"Large value result: {result.tolist()}")
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
```

<details>

<summary>
OverflowError: Python int too large to convert to C long
</summary>
```
Small values work: [Timestamp('1960-01-01 00:00:00'), Timestamp('1960-01-01 00:01:40'), Timestamp('1960-01-01 00:16:40')]
Error type: OverflowError
Error message: Python int too large to convert to C long
```
</details>

## Why This Is A Bug

This violates expected behavior for several reasons:

1. **Inconsistent error handling**: While the function gracefully handles NaN values in normal operation (they are preserved), it crashes completely on overflow values rather than degrading gracefully to NaT.

2. **Violates pandas conventions**: Throughout the pandas library, invalid datetime conversions typically return NaT rather than raising exceptions when handling data. For example, `pd.to_datetime()` with `errors='coerce'` converts invalid dates to NaT.

3. **Data import robustness**: The function is part of the SAS file reading pipeline, where corrupted or extreme values may be encountered. A single extreme value shouldn't prevent reading the entire file.

4. **Legitimate float64 values cause crashes**: The failing value `9.223372036854776e+18` is a valid float64 number that could appear in SAS files, especially if data is corrupted or misinterpreted.

## Relevant Context

The `_convert_datetimes` function is a vectorized datetime converter used when reading SAS7BDAT files. It converts SAS datetime values (stored as floats representing seconds or days since 1960-01-01) to pandas Timestamp objects.

The function uses `cast_from_unit_vectorized` from pandas' internal Cython library, which performs low-level conversion but doesn't handle overflow cases. When values represent dates millions of years in the future (like the failing input which is about 292 billion years from 1960), the conversion to milliseconds overflows C long integer limits.

Related code locations:
- Function definition: `/pandas/io/sas/sas7bdat.py:81-108`
- Called by: SAS file readers when processing datetime columns
- Similar function: `_parse_datetime` (lines 67-78) handles scalar values

Documentation references:
- SAS datetime format: Values represent seconds (unit='s') or days (unit='d') since January 1, 1960
- Pandas datetime limits: Timestamps are limited to approximately year 1677 to 2262 due to nanosecond precision

## Proposed Fix

```diff
--- a/pandas/io/sas/sas7bdat.py
+++ b/pandas/io/sas/sas7bdat.py
@@ -98,12 +98,22 @@ def _convert_datetimes(sas_datetimes: pd.Series, unit: str) -> pd.Series:
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
+            # Return NaT for values that overflow
+            return pd.Series([pd.NaT] * len(sas_datetimes), index=sas_datetimes.index)
     else:
-        vals = np.array(sas_datetimes, dtype="M8[D]") + td
-        return pd.Series(vals, dtype="M8[s]", index=sas_datetimes.index, copy=False)
+        try:
+            vals = np.array(sas_datetimes, dtype="M8[D]") + td
+            return pd.Series(vals, dtype="M8[s]", index=sas_datetimes.index, copy=False)
+        except (OverflowError, ValueError, pd._libs.tslibs.np_datetime.OutOfBoundsDatetime):
+            # Return NaT for values that overflow
+            return pd.Series([pd.NaT] * len(sas_datetimes), index=sas_datetimes.index)
```