# Bug Report: timedelta_isoformat Incorrect Encoding for Negative Timedeltas

**Target**: `pydantic.deprecated.json.timedelta_isoformat`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `timedelta_isoformat` function incorrectly encodes negative timedeltas when the timedelta has both negative days and positive seconds components (due to Python's normalization). The function applies a negative sign to the entire ISO 8601 string but uses the positive `td.seconds` value, resulting in an ISO format that represents a different duration than the input timedelta.

## Property-Based Test

```python
import datetime
from hypothesis import given, strategies as st, settings
from pydantic.deprecated.json import timedelta_isoformat


@given(st.timedeltas(
    min_value=datetime.timedelta(days=-999),
    max_value=datetime.timedelta(days=999)
))
@settings(max_examples=1000)
def test_timedelta_isoformat_total_seconds_consistency(td):
    iso_str = timedelta_isoformat(td)

    is_negative = iso_str.startswith('-P')

    parts = iso_str.replace('-P', 'P').split('DT')
    days_part = parts[0].replace('P', '').replace('D', '')
    time_part = parts[1]

    days = int(days_part) if days_part else 0

    hours_idx = time_part.index('H')
    hours = int(time_part[:hours_idx])

    minutes_start = hours_idx + 1
    minutes_idx = time_part.index('M', minutes_start)
    minutes = int(time_part[minutes_start:minutes_idx])

    seconds_start = minutes_idx + 1
    seconds_idx = time_part.index('.', seconds_start)
    seconds = int(time_part[seconds_start:seconds_idx])

    microseconds_start = seconds_idx + 1
    microseconds_end = time_part.index('S', microseconds_start)
    microseconds = int(time_part[microseconds_start:microseconds_end])

    computed_total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds + microseconds / 1000000
    if is_negative:
        computed_total_seconds = -computed_total_seconds

    actual_total_seconds = td.total_seconds()

    assert abs(computed_total_seconds - actual_total_seconds) < 0.000001
```

**Failing input**: `datetime.timedelta(days=-2, hours=2, minutes=30)` (or any negative timedelta with positive time components)

## Reproducing the Bug

```python
import datetime
from pydantic.deprecated.json import timedelta_isoformat

td = datetime.timedelta(days=-2, hours=2, minutes=30)

print(f"Input timedelta: {td}")
print(f"  total_seconds(): {td.total_seconds()}")
print(f"  This is {td.total_seconds() / 3600} hours")

iso = timedelta_isoformat(td)
print(f"\nISO format output: {iso}")
print(f"  This represents: -(2 days + 2 hours + 30 minutes)")
print(f"  Which equals: -50.5 hours = {-50.5 * 3600} seconds")

print(f"\nExpected: {td.total_seconds()} seconds")
print(f"Actual (from ISO): {-50.5 * 3600} seconds")
print(f"Error: {-50.5 * 3600 - td.total_seconds()} seconds")
```

Output:
```
Input timedelta: -2 days, 2:30:00
  total_seconds(): -163800.0
  This is -45.5 hours

ISO format output: -P2DT2H30M0.000000S
  This represents: -(2 days + 2 hours + 30 minutes)
  Which equals: -50.5 hours = -181800.0 seconds

Expected: -163800.0 seconds
Actual (from ISO): -181800.0 seconds
Error: -18000.0 seconds
```

## Why This Is A Bug

Python's `timedelta` normalizes negative durations such that `seconds` is always in the range [0, 86400). For `timedelta(days=-2, hours=2, minutes=30)`, Python normalizes this to `days=-2, seconds=9000`. This represents -2 days + 2.5 hours = -45.5 hours total.

However, `timedelta_isoformat` takes this normalized representation and outputs `-P2DT2H30M0.000000S`, which in ISO 8601 represents -(2 days + 2 hours + 30 minutes) = -50.5 hours. This is incorrect.

The bug violates the fundamental property that the ISO format should represent the same duration as the input timedelta.

## Fix

The fix needs to handle Python's timedelta normalization for negative durations. Instead of using the normalized `td.days`, `td.seconds`, and `td.microseconds` directly, we should use `td.total_seconds()` and decompose from there:

```diff
 def timedelta_isoformat(td: datetime.timedelta) -> str:
     """ISO 8601 encoding for Python timedelta object."""
     warnings.warn('`timedelta_isoformat` is deprecated.', category=PydanticDeprecatedSince20, stacklevel=2)
-    minutes, seconds = divmod(td.seconds, 60)
-    hours, minutes = divmod(minutes, 60)
-    return f'{"-" if td.days < 0 else ""}P{abs(td.days)}DT{hours:d}H{minutes:d}M{seconds:d}.{td.microseconds:06d}S'
+    total_seconds = td.total_seconds()
+    sign = '-' if total_seconds < 0 else ''
+    total_seconds = abs(total_seconds)
+
+    microseconds = int((total_seconds % 1) * 1000000)
+    total_seconds = int(total_seconds)
+
+    days, remainder = divmod(total_seconds, 86400)
+    hours, remainder = divmod(remainder, 3600)
+    minutes, seconds = divmod(remainder, 60)
+
+    return f'{sign}P{days}DT{hours}H{minutes}M{seconds}.{microseconds:06d}S'
```