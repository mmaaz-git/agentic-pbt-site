# Bug Report: pydantic.deprecated.json.timedelta_isoformat Incorrect ISO 8601 Encoding for Negative Timedeltas

**Target**: `pydantic.deprecated.json.timedelta_isoformat`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `timedelta_isoformat` function produces incorrect ISO 8601 duration strings for negative timedeltas that have non-zero seconds or microseconds components. The function incorrectly applies the negative sign to all components instead of properly handling Python's internal timedelta representation.

## Property-Based Test

```python
import datetime
import re
from hypothesis import given, strategies as st
from pydantic.deprecated.json import timedelta_isoformat


def parse_iso_duration(iso_string):
    match = re.match(r'^(-?)P(\d+)DT(\d+)H(\d+)M(\d+)\.(\d+)S$', iso_string)
    if not match:
        raise ValueError(f"Invalid ISO duration: {iso_string}")

    negative = match.group(1) == '-'
    days = int(match.group(2))
    hours = int(match.group(3))
    minutes = int(match.group(4))
    seconds = int(match.group(5))
    microseconds = int(match.group(6))

    total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds + microseconds / 1_000_000

    if negative:
        total_seconds = -total_seconds

    return total_seconds


@given(st.timedeltas(
    min_value=datetime.timedelta(days=-999999),
    max_value=datetime.timedelta(days=999999)
))
def test_timedelta_isoformat_roundtrip_value(td):
    iso_str = timedelta_isoformat(td)

    reconstructed_seconds = parse_iso_duration(iso_str)
    original_seconds = td.total_seconds()

    assert abs(reconstructed_seconds - original_seconds) < 1e-6, (
        f"ISO format doesn't preserve value!\n"
        f"  Original: {td} ({original_seconds} seconds)\n"
        f"  ISO: {iso_str}\n"
        f"  Reconstructed: {reconstructed_seconds} seconds\n"
        f"  Difference: {reconstructed_seconds - original_seconds} seconds"
    )
```

**Failing input**: `datetime.timedelta(days=-1, microseconds=1)`

## Reproducing the Bug

```python
import datetime
from pydantic.deprecated.json import timedelta_isoformat

td = datetime.timedelta(days=-1, microseconds=1)

print(f"Timedelta: {td}")
print(f"Total seconds: {td.total_seconds()}")

iso_str = timedelta_isoformat(td)
print(f"ISO format: {iso_str}")

print(f"\nExpected: -86399.999999 seconds")
print(f"ISO string represents: -(1 day + 0.000001 sec) = -86400.000001 seconds")
print(f"Difference: {-86400.000001 - (-86399.999999)} seconds")
```

## Why This Is A Bug

Python's internal representation of negative timedeltas uses a negative `days` component and positive `seconds` and `microseconds` components for the remainder. For example, `timedelta(days=-1, microseconds=1)` represents -1 day + 1 microsecond = -86399.999999 seconds.

However, the current implementation produces `-P1DT0H0M0.000001S`, which according to ISO 8601 means -(1 day + 0 hours + 0 minutes + 0.000001 seconds) = -86400.000001 seconds. This is off by 2 microseconds.

The bug occurs because the function applies the negative sign to the entire duration string but uses the raw `td.seconds` and `td.microseconds` values, which are always non-negative in Python's timedelta representation.

## Fix

```diff
 def timedelta_isoformat(td: datetime.timedelta) -> str:
-    minutes, seconds = divmod(td.seconds, 60)
-    hours, minutes = divmod(minutes, 60)
-    return f'{"-" if td.days < 0 else ""}P{abs(td.days)}DT{hours:d}H{minutes:d}M{seconds:d}.{td.microseconds:06d}S'
+    total_seconds = td.total_seconds()
+    negative = total_seconds < 0
+
+    if negative:
+        td = -td
+
+    days = td.days
+    hours, remainder = divmod(td.seconds, 3600)
+    minutes, seconds = divmod(remainder, 60)
+    microseconds = td.microseconds
+
+    return f'{"-" if negative else ""}P{days}DT{hours}H{minutes}M{seconds}.{microseconds:06d}S'
```