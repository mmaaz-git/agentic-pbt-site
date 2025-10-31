# Bug Report: xarray.compat.pdcompat Timestamp Functions Crash on Historical Dates

**Target**: `xarray.compat.pdcompat.timestamp_as_unit` and `xarray.compat.pdcompat.default_precision_timestamp`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Both `timestamp_as_unit` and `default_precision_timestamp` crash with `OutOfBoundsDatetime` when given valid timestamps outside the nanosecond-representable range (years before 1678 or after 2262), failing to handle a known pandas limitation gracefully.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st
from xarray.compat.pdcompat import timestamp_as_unit, default_precision_timestamp


@given(
    st.integers(min_value=1, max_value=9999),
    st.integers(min_value=1, max_value=12),
    st.integers(min_value=1, max_value=28),
    st.sampled_from(['s', 'ms', 'us', 'ns'])
)
def test_timestamp_as_unit_preserves_value(year, month, day, unit):
    ts = pd.Timestamp(year=year, month=month, day=day)
    result = timestamp_as_unit(ts, unit)
    assert result.year == ts.year
    assert result.month == ts.month
    assert result.day == ts.day


@given(
    st.integers(min_value=1, max_value=9999),
    st.integers(min_value=1, max_value=12),
    st.integers(min_value=1, max_value=28)
)
def test_default_precision_timestamp_returns_ns(year, month, day):
    result = default_precision_timestamp(year=year, month=month, day=day)
    assert result.unit == "ns"
```

<details>

<summary>
**Failing input**: `year=1, month=1, day=1, unit='ns'`
</summary>
```
Testing test_timestamp_as_unit_preserves_value:
------------------------------------------------------------
Trying example: test_timestamp_as_unit_preserves_value(
    year=1,
    month=1,
    day=1,
    unit='ns',
)
Traceback (most recent call last):
  File "pandas/_libs/tslibs/timestamps.pyx", line 1075, in pandas._libs.tslibs.timestamps._Timestamp._as_creso
  File "pandas/_libs/tslibs/np_datetime.pyx", line 683, in pandas._libs.tslibs.np_datetime.convert_reso
OverflowError: result would overflow

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo_verbose.py", line 15, in test_timestamp_as_unit_preserves_value
    result = timestamp_as_unit(ts, unit)
  File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/compat/pdcompat.py", line 84, in timestamp_as_unit
    date = date.as_unit(unit)
  File "pandas/_libs/tslibs/timestamps.pyx", line 1114, in pandas._libs.tslibs.timestamps._Timestamp.as_unit
  File "pandas/_libs/tslibs/timestamps.pyx", line 1078, in pandas._libs.tslibs.timestamps._Timestamp._as_creso
pandas._libs.tslibs.np_datetime.OutOfBoundsDatetime: Cannot cast 0001-01-01 00:00:00 to unit='ns' without overflow.

Test failed!

Testing test_default_precision_timestamp_returns_ns:
------------------------------------------------------------
Trying example: test_default_precision_timestamp_returns_ns(
    year=1,
    month=1,
    day=1,
)
Traceback (most recent call last):
  File "pandas/_libs/tslibs/timestamps.pyx", line 1075, in pandas._libs.tslibs.timestamps._Timestamp._as_creso
  File "pandas/_libs/tslibs/np_datetime.pyx", line 683, in pandas._libs.tslibs.np_datetime.convert_reso
OverflowError: result would overflow

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo_verbose.py", line 28, in test_default_precision_timestamp_returns_ns
    result = default_precision_timestamp(year=year, month=month, day=day)
  File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/compat/pdcompat.py", line 97, in default_precision_timestamp
    dt = timestamp_as_unit(dt, "ns")
  File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/compat/pdcompat.py", line 84, in timestamp_as_unit
    date = date.as_unit(unit)
  File "pandas/_libs/tslibs/timestamps.pyx", line 1114, in pandas._libs.tslibs.timestamps._Timestamp.as_unit
  File "pandas/_libs/tslibs/timestamps.pyx", line 1078, in pandas._libs.tslibs.timestamps._Timestamp._as_creso
pandas._libs.tslibs.np_datetime.OutOfBoundsDatetime: Cannot cast 0001-01-01 00:00:00 to unit='ns' without overflow.

Test failed!
```
</details>

## Reproducing the Bug

```python
import pandas as pd
from xarray.compat.pdcompat import timestamp_as_unit, default_precision_timestamp

print("Testing timestamp_as_unit with historical date (year=1000):")
print("-" * 60)
try:
    ts = pd.Timestamp(year=1000, month=1, day=1)
    print(f"Created pd.Timestamp: {ts}")
    print(f"Timestamp unit: {ts.unit}")
    result = timestamp_as_unit(ts, 'ns')
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\n")
print("Testing default_precision_timestamp with historical date (year=1000):")
print("-" * 60)
try:
    result = default_precision_timestamp(year=1000, month=1, day=1)
    print(f"Result: {result}")
    print(f"Result unit: {result.unit}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
```

<details>

<summary>
Output showing `OutOfBoundsDatetime` crash
</summary>
```
Testing timestamp_as_unit with historical date (year=1000):
------------------------------------------------------------
Created pd.Timestamp: 1000-01-01 00:00:00
Timestamp unit: us
Error: OutOfBoundsDatetime: Cannot cast 1000-01-01 00:00:00 to unit='ns' without overflow.


Testing default_precision_timestamp with historical date (year=1000):
------------------------------------------------------------
Error: OutOfBoundsDatetime: Cannot cast 1000-01-01 00:00:00 to unit='ns' without overflow.
```
</details>

## Why This Is A Bug

1. **Valid inputs crash**: The functions accept valid `pd.Timestamp` objects as input but crash instead of handling the limitation gracefully. Pandas successfully creates timestamps for year=1000 using microsecond precision ('us'), but these xarray compatibility functions fail when trying to convert to nanosecond precision.

2. **No documented limitations**: Neither function's docstring mentions that they only work with timestamps in the nanosecond-representable range (approximately 1678-2262). Users have no warning about this restriction.

3. **No input validation or error handling**: The functions directly call `as_unit()` without checking if the conversion is possible or catching the predictable `OutOfBoundsDatetime` exception.

4. **Breaks xarray's datetime functionality**: These functions are used in `xarray/coding/times.py` and `xarray/coding/cftime_offsets.py` for datetime encoding/decoding and calendar operations. Their failure can break xarray operations on datasets with historical dates.

5. **Inconsistent with pandas flexibility**: Pandas allows timestamps outside the ns range by automatically using appropriate units ('s', 'ms', 'us'). These compatibility functions force everything to 'ns', removing this flexibility.

6. **`default_precision_timestamp` cannot fulfill its contract**: The function promises to return a timestamp with "ns" precision, but this is impossible for dates outside the nanosecond range. The function should either handle this gracefully or document the limitation.

## Relevant Context

The nanosecond precision limitation in pandas is well-documented:
- Nanosecond timestamps can only represent dates from approximately 1677-09-21 to 2262-04-11
- Pandas automatically uses microsecond precision for dates outside this range
- The `as_unit()` method correctly raises `OutOfBoundsDatetime` when conversion would overflow

These xarray compatibility functions are marked as temporary (to be removed when minimum pandas version is >= 2.2), but they're currently used throughout xarray's codebase for critical datetime operations. Scientific datasets commonly include historical climate data, geological records, and astronomical observations that use dates well outside the nanosecond range.

## Proposed Fix

The functions should catch the `OutOfBoundsDatetime` exception and handle it gracefully:

```diff
--- a/xarray/compat/pdcompat.py
+++ b/xarray/compat/pdcompat.py
@@ -76,11 +76,21 @@ def timestamp_as_unit(date: pd.Timestamp, unit: PDDatetimeUnitOptions) -> pd.Ti
     """Convert the underlying int64 representation to the given unit.

     Compatibility function for pandas issue where "as_unit" is not defined
     for pandas.Timestamp in pandas versions < 2.2. Can be removed minimum
     pandas version is >= 2.2.
+
+    Note: Conversion may fail for timestamps outside the representable range
+    of the target unit. In particular, 'ns' can only represent years ~1678-2262.
     """
-    if hasattr(date, "as_unit"):
-        date = date.as_unit(unit)
-    elif hasattr(date, "_as_unit"):
-        date = date._as_unit(unit)
+    try:
+        if hasattr(date, "as_unit"):
+            date = date.as_unit(unit)
+        elif hasattr(date, "_as_unit"):
+            date = date._as_unit(unit)
+    except pd._libs.tslibs.np_datetime.OutOfBoundsDatetime:
+        # If conversion fails, return the original timestamp
+        # This preserves the timestamp value even if it can't be represented
+        # in the requested unit
+        pass
     return date


@@ -90,8 +100,14 @@ def default_precision_timestamp(*args, **kwargs) -> pd.Timestamp:
     """Return a Timestamp object with the default precision.

     Xarray default is "ns".
+
+    Note: For timestamps outside the ns-representable range (~1678-2262),
+    returns a timestamp with the best available precision instead.
     """
     dt = pd.Timestamp(*args, **kwargs)
-    if dt.unit != "ns":
-        dt = timestamp_as_unit(dt, "ns")
+    try:
+        if dt.unit != "ns":
+            dt = timestamp_as_unit(dt, "ns")
+    except pd._libs.tslibs.np_datetime.OutOfBoundsDatetime:
+        pass  # Keep the original unit if ns conversion fails
     return dt
```