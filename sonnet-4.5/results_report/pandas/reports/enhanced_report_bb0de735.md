# Bug Report: pandas.io.sas.sas7bdat._parse_datetime OverflowError on Large SAS Date Values

**Target**: `pandas.io.sas.sas7bdat._parse_datetime`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `_parse_datetime` function crashes with an OverflowError when processing SAS datetime values that would result in dates beyond Python's datetime.max year (9999), preventing pandas from reading valid SAS files containing far-future dates.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.io.sas.sas7bdat import _parse_datetime
from datetime import datetime
import pandas as pd


@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10))
@settings(max_examples=1000)
def test_parse_datetime_days_unit(sas_datetime):
    result = _parse_datetime(sas_datetime, unit='d')

    if not pd.isna(result):
        assert isinstance(result, datetime)


if __name__ == "__main__":
    test_parse_datetime_days_unit()
```

<details>

<summary>
**Failing input**: `sas_datetime=2936550.0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 17, in <module>
    test_parse_datetime_days_unit()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 8, in test_parse_datetime_days_unit
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 10, in test_parse_datetime_days_unit
    result = _parse_datetime(sas_datetime, unit='d')
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/sas/sas7bdat.py", line 75, in _parse_datetime
    return datetime(1960, 1, 1) + timedelta(days=sas_datetime)
           ~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
OverflowError: date value out of range
Falsifying example: test_parse_datetime_days_unit(
    sas_datetime=2936550.0,
)
```
</details>

## Reproducing the Bug

```python
from pandas.io.sas.sas7bdat import _parse_datetime
from datetime import datetime

# Test the exact failing input from the bug report
sas_datetime = 2936550.0

try:
    result = _parse_datetime(sas_datetime, unit='d')
    print(f"Result: {result}")
except OverflowError as e:
    print(f"OverflowError: {e}")

# Also test the boundary case
print("\nBoundary testing:")
print("Testing 2936549.0 (should work):")
try:
    result = _parse_datetime(2936549.0, unit='d')
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")

print("\nTesting 2936550.0 (should fail):")
try:
    result = _parse_datetime(2936550.0, unit='d')
    print(f"Result: {result}")
except Exception as e:
    print(f"Error type: {type(e).__name__}: {e}")
```

<details>

<summary>
OverflowError: date value out of range
</summary>
```
OverflowError: date value out of range

Boundary testing:
Testing 2936549.0 (should work):
Result: 9999-12-31 00:00:00

Testing 2936550.0 (should fail):
Error type: OverflowError: date value out of range
```
</details>

## Why This Is A Bug

This violates expected behavior because:

1. **SAS supports wider date ranges**: SAS can legitimately store dates from 1582 AD to approximately year 20,000 AD, while Python's datetime is limited to years 1-9999. The value 2936550 days from 1960-01-01 represents January 1, 10000, which is valid in SAS.

2. **Inconsistent error handling**: The function already handles unparseable values (NaN) gracefully by returning `pd.NaT` (lines 68-69), establishing a pattern for dealing with values that cannot be represented. However, it doesn't extend this pattern to handle Python's datetime limitations.

3. **Prevents file reading**: When reading SAS files containing even a single date beyond year 9999, the entire read operation crashes with an unhandled OverflowError, making it impossible to import the data.

4. **Sister function suggests graceful handling**: The `_convert_datetimes` function's docstring states "Convert to Timestamp if possible", implying conversion should degrade gracefully when not possible rather than crashing.

5. **No documented limitation**: Neither the function nor the SAS7BDATReader class documentation mentions this limitation, leaving users unprepared for the crash.

## Relevant Context

The exact boundary is at 2936549 days from January 1, 1960:
- `_parse_datetime(2936549.0, 'd')` returns `datetime(9999, 12, 31)` - the maximum Python datetime
- `_parse_datetime(2936550.0, 'd')` raises OverflowError - would be year 10000

The function is called during SAS file reading in the `_chunk_to_dataframe` method (lines 734-738 in sas7bdat.py) when `convert_dates=True` and the column format indicates a SAS date or datetime format. This makes the crash particularly problematic as it occurs during normal file reading operations.

Similar overflow issues can occur with:
- Large negative values (dates before year 1)
- Very large values that exceed timedelta limits (|days| > 999999999)
- Extreme values that cause integer overflow when converting to C int

SAS date/datetime documentation: https://documentation.sas.com/doc/en/vdmmlcdc/1.0/ds2ref/n0n3hquw7n3p1gn11t5t8ymvzm8a.htm

## Proposed Fix

```diff
--- a/pandas/io/sas/sas7bdat.py
+++ b/pandas/io/sas/sas7bdat.py
@@ -67,13 +67,21 @@ _sas_origin = Timestamp("1960-01-01")
 def _parse_datetime(sas_datetime: float, unit: str):
     if isna(sas_datetime):
         return pd.NaT

     if unit == "s":
-        return datetime(1960, 1, 1) + timedelta(seconds=sas_datetime)
+        try:
+            return datetime(1960, 1, 1) + timedelta(seconds=sas_datetime)
+        except (OverflowError, OSError):
+            return pd.NaT

     elif unit == "d":
-        return datetime(1960, 1, 1) + timedelta(days=sas_datetime)
+        try:
+            return datetime(1960, 1, 1) + timedelta(days=sas_datetime)
+        except (OverflowError, OSError):
+            return pd.NaT

     else:
         raise ValueError("unit must be 'd' or 's'")
```