# Bug Report: xarray.compat.pdcompat.default_precision_timestamp Overflow on Dates Beyond Nanosecond Range

**Target**: `xarray.compat.pdcompat.default_precision_timestamp`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `default_precision_timestamp` function crashes with an `OutOfBoundsDatetime` exception when given datetime objects beyond the nanosecond precision range (dates after 2262-04-11), despite pandas being able to handle these dates using alternative precision units.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from xarray.compat.pdcompat import default_precision_timestamp


@given(st.datetimes())
@settings(max_examples=1000)
def test_default_precision_timestamp_unit(dt):
    result = default_precision_timestamp(dt)
    assert result.unit == 'ns'

if __name__ == "__main__":
    test_default_precision_timestamp_unit()
```

<details>

<summary>
**Failing input**: `datetime.datetime(2263, 1, 1, 0, 0)`
</summary>
```
Traceback (most recent call last):
  File "pandas/_libs/tslibs/timestamps.pyx", line 1075, in pandas._libs.tslibs.timestamps._Timestamp._as_creso
  File "pandas/_libs/tslibs/np_datetime.pyx", line 683, in pandas._libs.tslibs.np_datetime.convert_reso
OverflowError: result would overflow

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 15, in <module>
    test_default_precision_timestamp_unit()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 9, in test_default_precision_timestamp_unit
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 11, in test_default_precision_timestamp_unit
    result = default_precision_timestamp(dt)
  File "/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/xarray/compat/pdcompat.py", line 97, in default_precision_timestamp
    dt = timestamp_as_unit(dt, "ns")
  File "/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/xarray/compat/pdcompat.py", line 84, in timestamp_as_unit
    date = date.as_unit(unit)
  File "pandas/_libs/tslibs/timestamps.pyx", line 1114, in pandas._libs.tslibs.timestamps._Timestamp.as_unit
  File "pandas/_libs/tslibs/timestamps.pyx", line 1078, in pandas._libs.tslibs.timestamps._Timestamp._as_creso
pandas._libs.tslibs.np_datetime.OutOfBoundsDatetime: Cannot cast 2263-01-01 00:00:00 to unit='ns' without overflow.
Falsifying example: test_default_precision_timestamp_unit(
    dt=datetime.datetime(2263, 1, 1, 0, 0),
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import datetime
import pandas as pd
from xarray.compat.pdcompat import default_precision_timestamp

# Create a datetime beyond nanosecond precision range (after 2262-04-11)
dt = datetime.datetime(2263, 1, 1, 0, 0)

# Show that pd.Timestamp can handle this date
ts = pd.Timestamp(dt)
print(f"pd.Timestamp works: {ts}, unit={ts.unit}")

# Now try with default_precision_timestamp - this will crash
try:
    result = default_precision_timestamp(dt)
    print(f"default_precision_timestamp result: {result}, unit={result.unit}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
```

<details>

<summary>
OutOfBoundsDatetime: Cannot cast 2263-01-01 00:00:00 to unit='ns' without overflow
</summary>
```
pd.Timestamp works: 2263-01-01 00:00:00, unit=us
Error: OutOfBoundsDatetime: Cannot cast 2263-01-01 00:00:00 to unit='ns' without overflow.
```
</details>

## Why This Is A Bug

This function violates its implicit contract of being a wrapper around `pd.Timestamp` that adds default precision handling. The bug manifests because:

1. **Pandas handles dates beyond 2262-04-11 correctly**: When creating a `pd.Timestamp` from a datetime beyond the nanosecond range, pandas automatically selects an appropriate precision unit (microseconds 'us' in this case) that can represent the date without overflow.

2. **The function blindly forces nanosecond precision**: The `default_precision_timestamp` function unconditionally attempts to convert all timestamps to nanosecond precision on line 97, without checking if this conversion is possible.

3. **Function signature implies compatibility**: The function accepts `*args, **kwargs` matching `pd.Timestamp`'s signature, suggesting it should handle all valid inputs that `pd.Timestamp` accepts.

4. **Real-world impact**: This function is used in critical xarray components including `xarray.coding.times` (line 861) and `xarray.coding.cftime_offsets` (lines 92, 1611-1612), affecting time encoding/decoding operations for datasets with dates beyond 2262.

5. **Documentation contradiction**: The docstring states "Xarray default is 'ns'" but doesn't warn about or handle the overflow case, despite pandas itself gracefully handling these dates with alternative units.

## Relevant Context

The nanosecond precision limit in pandas timestamps is well-documented. The maximum representable date with nanosecond precision is approximately 2262-04-11 23:47:16.854775807. Pandas addressed this limitation by introducing multiple precision units (seconds 's', milliseconds 'ms', microseconds 'us', nanoseconds 'ns').

Key code locations:
- Function definition: `/xarray/compat/pdcompat.py:90-98`
- Helper function `timestamp_as_unit`: `/xarray/compat/pdcompat.py:76-87`
- Usage in time encoding: `/xarray/coding/times.py:861`
- Usage in cftime offsets: `/xarray/coding/cftime_offsets.py:92,1611-1612`

This bug would affect scientific computing applications dealing with:
- Long-term climate projections
- Astronomical data with future dates
- Financial modeling with extended time horizons
- Any dataset requiring dates beyond year 2262

## Proposed Fix

```diff
--- a/xarray/compat/pdcompat.py
+++ b/xarray/compat/pdcompat.py
@@ -90,9 +90,16 @@ def timestamp_as_unit(date: pd.Timestamp, unit: PDDatetimeUnitOptions) -> pd.Ti
 def default_precision_timestamp(*args, **kwargs) -> pd.Timestamp:
     """Return a Timestamp object with the default precision.

-    Xarray default is "ns".
+    Xarray default is "ns" when possible, but preserves the timestamp's
+    original unit for dates outside the nanosecond range.
     """
     dt = pd.Timestamp(*args, **kwargs)
     if dt.unit != "ns":
-        dt = timestamp_as_unit(dt, "ns")
+        try:
+            dt = timestamp_as_unit(dt, "ns")
+        except (OverflowError, pd._libs.tslibs.np_datetime.OutOfBoundsDatetime):
+            # Keep the timestamp at its current unit if conversion to ns would overflow
+            # This happens for dates beyond 2262-04-11 (the max nanosecond timestamp)
+            pass
     return dt
```