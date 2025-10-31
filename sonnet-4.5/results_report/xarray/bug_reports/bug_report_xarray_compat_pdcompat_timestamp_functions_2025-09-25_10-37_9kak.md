# Bug Report: xarray.compat.pdcompat Timestamp Functions Crash on Historical Dates

**Target**: `xarray.compat.pdcompat.timestamp_as_unit` and `xarray.compat.pdcompat.default_precision_timestamp`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Both `timestamp_as_unit` and `default_precision_timestamp` crash with `OutOfBoundsDatetime` when given valid timestamps outside the nanosecond-representable range (years before 1678 or after 2262). The functions don't validate inputs or handle this limitation gracefully.

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

**Failing input**: `year=1, month=1, day=1` (or any year before 1678)

## Reproducing the Bug

```python
import pandas as pd
from xarray.compat.pdcompat import timestamp_as_unit, default_precision_timestamp

ts = pd.Timestamp(year=1000, month=1, day=1)
timestamp_as_unit(ts, 'ns')

default_precision_timestamp(year=1000, month=1, day=1)
```

Both calls raise:
```
pandas._libs.tslibs.np_datetime.OutOfBoundsDatetime: Cannot cast 1000-01-01 00:00:00 to unit='ns' without overflow.
```

## Why This Is A Bug

1. **Valid inputs crash the functions**: Timestamps with year=1000 are valid `pd.Timestamp` objects, but these functions crash instead of handling the limitation gracefully.

2. **No documentation of limitations**: Neither function's docstring mentions that they only work with timestamps in the range 1678-2262 (the ns-representable range).

3. **No input validation**: The functions could check if conversion to 'ns' is possible before attempting it, especially since `default_precision_timestamp` unconditionally tries to convert to 'ns'.

4. **Inconsistent with pandas behavior**: Pandas allows creating timestamps outside the ns range by using different units (like 'us' or 'ms'). These compatibility functions break that flexibility.

## Fix

The functions should either:
1. Validate that the timestamp can be represented in the target unit before conversion, or
2. Catch the `OutOfBoundsDatetime` exception and handle it gracefully (e.g., return the original timestamp if conversion fails)

Here's a proposed fix:

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