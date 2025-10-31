# Bug Report: xarray.compat.pdcompat default_precision_timestamp Overflow

**Target**: `xarray.compat.pdcompat.default_precision_timestamp`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `default_precision_timestamp` function crashes when given a datetime beyond the nanosecond precision range (after 2262-04-11), even though `pd.Timestamp` can handle such dates using higher precision units like microseconds or seconds.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from xarray.compat.pdcompat import default_precision_timestamp


@given(st.datetimes())
def test_default_precision_timestamp_unit(dt):
    result = default_precision_timestamp(dt)
    assert result.unit == 'ns'
```

**Failing input**: `datetime.datetime(2263, 1, 1, 0, 0)`

## Reproducing the Bug

```python
import datetime
import pandas as pd
from xarray.compat.pdcompat import default_precision_timestamp

dt = datetime.datetime(2263, 1, 1, 0, 0)

ts = pd.Timestamp(dt)
print(f"pd.Timestamp works: {ts}, unit={ts.unit}")

result = default_precision_timestamp(dt)
```

**Output**:
```
pd.Timestamp works: 2263-01-01 00:00:00, unit=us
pandas._libs.tslibs.np_datetime.OutOfBoundsDatetime: Cannot cast 2263-01-01 00:00:00 to unit='ns' without overflow.
```

## Why This Is A Bug

1. `pd.Timestamp` can handle dates beyond 2262-04-11 by using higher precision units (microseconds, seconds, etc.)
2. When `pd.Timestamp` is created from such a datetime, it automatically selects an appropriate unit (e.g., 'us')
3. `default_precision_timestamp` blindly tries to convert all timestamps to nanosecond precision, causing an overflow
4. The function accepts any arguments that `pd.Timestamp` accepts, but crashes on valid inputs that pandas can handle
5. This is used in xarray's time encoding/decoding (`xarray.coding.times` and `xarray.coding.cftime_offsets`), so it affects real use cases with dates outside the nanosecond range

## Fix

```diff
--- a/xarray/compat/pdcompat.py
+++ b/xarray/compat/pdcompat.py
@@ -90,9 +90,16 @@ def timestamp_as_unit(date: pd.Timestamp, unit: PDDatetimeUnitOptions) -> pd.Ti
 def default_precision_timestamp(*args, **kwargs) -> pd.Timestamp:
     """Return a Timestamp object with the default precision.

-    Xarray default is "ns".
+    Xarray default is "ns" when possible, but uses higher precision for
+    timestamps outside the nanosecond range.
     """
     dt = pd.Timestamp(*args, **kwargs)
-    if dt.unit != "ns":
-        dt = timestamp_as_unit(dt, "ns")
+    if dt.unit != "ns":
+        try:
+            dt = timestamp_as_unit(dt, "ns")
+        except (OverflowError, pd._libs.tslibs.np_datetime.OutOfBoundsDatetime):
+            # Keep the timestamp at its current unit if conversion to ns would overflow
+            # This happens for dates beyond 2262-04-11 (the max nanosecond timestamp)
+            pass
     return dt
```